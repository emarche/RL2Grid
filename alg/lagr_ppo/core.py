from time import time

from torch.nn.functional import softplus

from .agent import Agent
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import ConstrainedLogger
from env.eval import CMDPEvaluator

class LagrPPO:
    """Lagrangian implementation of the Lagrangian Proximal Policy Optimization (LagrPPO) implementation for training a constrained agent in a given environment: https://cdn.openai.com/safexp-short.pdf.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        """Init method for Lagrangian PPO

        Args:
            envs (gym.Env): The environments used for training.
            run_name (str): The name of the current training run.
            start_time (float): The time when training started.
            args (Dict[str, Any]): The command line arguments for configuration.
            ckpt (CheckpointSaver): The checkpoint handler for saving and loading training state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.n_steps % args.n_envs == 0, \
            f"Invalid train frequency (n_steps): {args.n_steps}. Must be multiple of n_envs {args.n_envs}"

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # Initialize the rollout, actor, critic, optimizer, and buffer
        batch_size = int(args.n_envs * args.n_steps)
        minibatch_size = int(batch_size // args.n_minibatches)
        n_rollouts = args.total_timesteps // batch_size
        init_rollout = 1 if not ckpt.resumed else ckpt.loaded_run['last_rollout']

        # Determine action space type
        continuous_actions = True if args.action_type == "redispatch" else False
        agent = Agent(envs, args, continuous_actions).to(device)

        if ckpt.resumed:
            agent.actor.load_state_dict(ckpt.loaded_run['actor'])
            agent.critic.load_state_dict(ckpt.loaded_run['critic'])
            agent.cost_critic.load_state_dict(ckpt.loaded_run['cost_critic'])

        actor_params = list(agent.actor.parameters())
        if continuous_actions: actor_params + [agent.logstd]
        actor_optim = optim.Adam(actor_params, lr=args.actor_lr, eps=1e-5)
        critic_optim = optim.Adam(agent.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        cost_critic_optim = optim.Adam(agent.cost_critic.parameters(), lr=args.critic_lr, eps=1e-5)

        # Lagrangian setup
        lag_mul = th.tensor(args.lag_mul, requires_grad=True, device=device)
        lag_optim = optim.Adam([lag_mul], lr=args.lag_lr)
        cost_threshold = args.cost_threshold

        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])
            cost_critic_optim.load_state_dict(ckpt.loaded_run['cost_critic_optim'])
            lag_mul = th.tensor(ckpt.loaded_run['lag_mul'], requires_grad=True, device=device)
            lag_optim.load_state_dict(ckpt.loaded_run['lag_optim'])
            cost_threshold = ckpt.loaded_run['cost_threshold']

        observations = th.zeros((args.n_steps, args.n_envs) + envs.single_observation_space.shape).to(device)
        actions = th.zeros((args.n_steps, args.n_envs) + 
                           envs.single_action_space.shape ).to(device)
        logprobs = th.zeros((args.n_steps, args.n_envs)).to(device)
        rewards = th.zeros((args.n_steps, args.n_envs)).to(device)
        costs = th.zeros((args.n_steps, args.n_envs)).to(device)
        dones = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        terminations = th.zeros((args.n_steps, args.n_envs), dtype=th.int32).to(device)
        values = th.zeros((args.n_steps, args.n_envs)).to(device)
        cost_values = th.zeros((args.n_steps, args.n_envs)).to(device)

        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"
        logger = ConstrainedLogger(run_name, args) if args.track else None
        evaluator = CMDPEvaluator(args, logger, device)

        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        next_obs, _ = envs.reset()
        next_obs = th.tensor(next_obs).to(device)

        try:
            for iteration in range(init_rollout, n_rollouts + 1):
                # Annealing the rate if instructed to do so
                if args.anneal_lr:
                    frac = 1.0 - (iteration - 1.0) / n_rollouts
                    actor_optim.param_groups[0]['lr'] = frac * args.actor_lr
                    critic_optim.param_groups[0]['lr'] = frac * args.critic_lr

                for step in range(0, args.n_steps):
                    global_step += args.n_envs
                    observations[step] = next_obs
                    
                    with th.no_grad():
                        action, logprob, _ = agent.get_action(next_obs)
                        value = agent.get_value(next_obs)
                        values[step] = value.flatten()
                        cost_value = agent.get_cost_value(next_obs)
                        cost_values[step] = cost_value.flatten()
                    actions[step] = action
                    logprobs[step] = logprob

                    next_obs, reward, next_terminations, next_truncations, infos = envs.step(action.cpu().numpy())

                    rewards[step] = th.tensor(reward).to(device).view(-1)
                    dones[step] = th.tensor(np.logical_or(next_terminations, next_truncations)).to(device)
                    terminations[step] = th.tensor(next_terminations).to(device)

                    real_next_obs = next_obs.copy()
                    cost = infos.get('cost', np.zeros(args.n_envs))
                    for idx, done in enumerate(dones[step]):
                        if done: 
                            real_next_obs[idx] = infos["final_observation"][idx]
                            cost[idx] =  infos["final_info"][idx]['cost']   # When done, 'cost' is also in 'final_info'
    
                    costs[step] = th.tensor(cost).to(device).view(-1)
                    next_obs = th.tensor(next_obs).to(device)
                    real_next_obs = th.tensor(real_next_obs).to(device)

                    if global_step % args.eval_freq == 0:
                        evaluator.evaluate(global_step, agent)
                        if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")

                # Lagrangian update
                costs_q = costs[-int(8000/args.n_envs):, :].flatten()
                lag_loss = -lag_mul * (th.mean(th.tensor(costs_q.sum(), dtype=th.float32)) - args.cost_threshold)

                lag_optim.zero_grad()
                lag_loss.backward()
                lag_optim.step()

                # Bootstrap value if not done
                with th.no_grad():
                    advantages = th.zeros_like(rewards).to(device)
                    cost_advantages = th.zeros_like(rewards).to(device)
                    lastgaelam, lastcostgaelam = 0, 0
                    for t in reversed(range(args.n_steps)):
                        if t == args.n_steps - 1:
                            nextvalues = agent.get_value(real_next_obs).reshape(1, -1)
                            nextcostvalues = agent.get_cost_value(real_next_obs).reshape(1, -1)
                        else:
                            nextvalues = values[t + 1]
                            nextcostvalues = cost_values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * (1 - terminations[t]) - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * (1 - dones[t]) * lastgaelam
                        cost_delta = costs[t] + args.gamma * nextcostvalues * (1 - terminations[t]) - cost_values[t]
                        cost_advantages[t] = lastcostgaelam = cost_delta + args.gamma * args.gae_lambda * (1 - dones[t]) * lastcostgaelam
                    returns = advantages + values
                    cost_returns = cost_advantages + cost_values
   
                # Flatten the batch
                b_obs = observations.reshape((-1,) + envs.single_observation_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_cost_advantages = cost_advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_cost_returns = cost_returns.reshape(-1)
                b_values = values.reshape(-1)
                b_cost_values = cost_values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(batch_size)
                clipfracs = []
                for _ in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        _, newlogprob, entropy = agent.get_action(b_obs[mb_inds], b_actions.long()[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with th.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            #old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        mb_cost_advantages = b_cost_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                            mb_cost_advantages -= mb_cost_advantages.mean()

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = th.max(pg_loss1, pg_loss2).mean()

                        # Cost component
                        surr_cost_advantage = (mb_cost_advantages * ratio).mean()
                        multiplier = softplus(lag_mul)

                        pg_loss = (pg_loss - multiplier * surr_cost_advantage) / (1 + multiplier)

                        entropy_loss = entropy.mean()
                        pg_loss = pg_loss - args.entropy_coef * entropy_loss

                        actor_optim.zero_grad()
                        pg_loss.backward()
                        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                        actor_optim.step()

                        # Value loss
                        newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                        newcostvalue = agent.get_cost_value(b_obs[mb_inds]).view(-1)
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
                            
                            cv_loss_unclipped = (newcostvalue - b_cost_returns[mb_inds]) ** 2
                            cv_clipped = b_cost_values[mb_inds] + th.clamp(
                                newcostvalue - b_cost_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            cv_loss_clipped = (cv_clipped - b_cost_returns[mb_inds]) ** 2
                            cv_loss_max = th.max(cv_loss_unclipped, cv_loss_clipped)
                            cv_loss = 0.5 * cv_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                            cv_loss = 0.5 * ((newcostvalue - b_cost_returns[mb_inds]) ** 2).mean()

                        v_loss *= args.vf_coef
                        cv_loss *= args.vf_coef

                        critic_optim.zero_grad()
                        v_loss.backward()
                        nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                        critic_optim.step()

                        cost_critic_optim.zero_grad()
                        cv_loss.backward()
                        nn.utils.clip_grad_norm_(agent.cost_critic.parameters(), args.max_grad_norm)
                        cost_critic_optim.step()

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                # If we reach the node's time limit, we just exit the training loop, save metrics and ckpt
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            # Save the checkpoint and logger data
            ckpt.set_record(args, agent.actor, agent.critic, agent.cost_critic, global_step, actor_optim, critic_optim, cost_critic_optim, cost_threshold, lag_mul, lag_optim, "" if not logger else logger.wb_path, iteration)
            ckpt.save()
            if logger: logger.close()
            envs.close()
