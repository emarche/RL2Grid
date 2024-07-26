from time import time

from .agent import Agent
from .config import get_alg_args
from common.imports import ap, nn, np, optim, th
from common.logger import Logger
from env.eval import Evaluator

class PPO:
    def __init__(self, envs, run_name, start_time, args, ckpt):        
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.n_steps % args.n_envs == 0, \
            f"Invalid train frequency (n_steps): {args.n_steps}. Must be multiple of n_envs {args.n_envs}"

        batch_size = int(args.n_envs * args.n_steps)
        minibatch_size = int(batch_size // args.n_minibatches)
        n_rollouts = args.total_timesteps // batch_size
        init_rollout = 1 if not ckpt.resumed else ckpt.loaded_run['last_rollout']

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        continuous_actions = True if args.action_type == "redispatch" else False
        agent = Agent(envs, args, continuous_actions).to(device)

        if ckpt.resumed:
            agent.actor.load_state_dict(ckpt.loaded_run['actor'])
            agent.critic.load_state_dict(ckpt.loaded_run['critic'])

        actor_params = list(agent.actor.parameters())
        if continuous_actions: actor_params + [agent.logstd]
        actor_optim = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(agent.critic.parameters(), lr=args.critic_lr, eps=1e-5)

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])

        # ALGO Logic: Storage setup
        observations = th.zeros((args.n_steps, args.n_envs) + envs.single_observation_space.shape).to(device)
        actions = th.zeros((args.n_steps, args.n_envs) + 
                           envs.single_action_space.shape ).to(device)
        logprobs = th.zeros((args.n_steps, args.n_envs)).to(device)
        rewards = th.zeros((args.n_steps, args.n_envs)).to(device)
        dones = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        values = th.zeros((args.n_steps, args.n_envs)).to(device)

        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"
        logger = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        next_obs, info = envs.reset()
        next_obs = th.tensor(next_obs).to(device)
        next_done = th.zeros(args.n_envs).to(device)

        try:
            for iteration in range(init_rollout, n_rollouts + 1):
                # Annealing the rate if instructed to do so.
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / n_rollouts
                    actor_optim.param_groups[0]['lr'] = frac * args.actor_lr
                    critic_optim.param_groups[0]['lr'] = frac * args.critic_lr

                for step in range(0, args.n_steps):
                    global_step += args.n_envs
                    observations[step] = next_obs
                    dones[step] = next_done

                    # ALGO LOGIC: action logic
                    with th.no_grad():
                        action, logprob, _ = agent.get_action(next_obs)
                        value = agent.get_value(next_obs)
                        values[step] = value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                
                    next_done = np.logical_or(terminations, truncations)
                    rewards[step] = th.tensor(reward).to(device).view(-1)
                    next_obs, next_done = th.tensor(next_obs).to(device), th.tensor(next_done).to(device)

                    if global_step % args.eval_freq == 0:
                        evaluator.evaluate(global_step, agent)

                # bootstrap value if not done
                with th.no_grad():
                    next_value = agent.get_value(next_obs).reshape(1, -1)
                    advantages = th.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.n_steps)):
                        if t == args.n_steps - 1:
                            nextnonterminal = ~next_done    # instead of (1 - next_done) just flip it
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]

                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values

                # flatten the batch
                b_obs = observations.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(batch_size)
                clipfracs = []
                for _ in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions.long()[mb_inds])
                        newvalue = agent.get_value(b_obs[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with th.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            #old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        
                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = th.max(pg_loss1, pg_loss2).mean()

                        entropy_loss = entropy.mean()
                        pg_loss = pg_loss - args.entropy_coef * entropy_loss

                        actor_optim.zero_grad()
                        pg_loss.backward()
                        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                        actor_optim.step()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if args.clip_vfloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + th.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        v_loss *= args.vf_coef

                        critic_optim.zero_grad()
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        critic_optim.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                if global_step % args.eval_freq == 0:
                    if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")
                
                # If we reach the node's time limit, we just exit the training loop, save metrics and ckpt
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            ckpt.set_record(args, agent.actor, agent.critic, global_step, actor_optim, critic_optim, "" if not logger else logger.wb_path, iteration)
            ckpt.save()
            if logger: logger.close()
            envs.close()
