from time import time

from stable_baselines3.common.buffers import ReplayBuffer

from .agent import Actor, SoftQNetwork
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator

class SAC:
    """Soft Actor-Critic (SAC) algorithm implementation: https://arxiv.org/abs/1801.01290.    
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        """Init method for SAC

        Args:
            envs (gym.Env): Environment object providing observation and action spaces.
            run_name (str): Name of the current run.
            start_time (float): Start time of the training process.
            args (Dict[str, Any]): Arguments containing various hyperparameters.
            ckpt (CheckpointSaver): Checkpoint object for saving and loading the model state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.train_freq % args.n_envs == 0, \
            f"Invalid train frequency: {args.train_freq}. Must be multiple of n_envs {args.n_envs}"
    
        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # Initialize the networks, optimizer, and buffer
        continuous_actions = True if args.action_type == "redispatch" else False
        actor = Actor(envs, args, continuous_actions).to(device)
        critic1 = SoftQNetwork(envs, args, continuous_actions).to(device)
        critic2 = SoftQNetwork(envs, args, continuous_actions).to(device)
        tg_critic1 = SoftQNetwork(envs, args, continuous_actions).to(device)
        tg_critic2 = SoftQNetwork(envs, args, continuous_actions).to(device)

        if ckpt.resumed:
            actor.load_state_dict(ckpt.loaded_run['actor'])
            critic1.load_state_dict(ckpt.loaded_run['critic'])
            critic2.load_state_dict(ckpt.loaded_run['critic2'])

        tg_critic1.load_state_dict(critic1.state_dict())
        tg_critic2.load_state_dict(critic2.state_dict())
        
        actor_optim = optim.Adam(list(actor.parameters()), lr=args.actor_lr)
        critic_optim = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=args.critic_lr)
        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])

        # Automatic entropy tuning
        if args.autotune:
            tg_entropy = -th.prod(th.tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = th.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            alpha_optim = optim.Adam([log_alpha], lr=args.critic_lr)
        else:
            alpha = args.alpha

        envs.single_observation_space.dtype = np.float32
        replay_buf = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.n_envs,
            handle_timeout_termination=False
        )

        assert args.eval_freq % args.n_envs == 0, \
            f"Invalid eval frequency: {args.eval_freq}. Must be multiple of n_envs {args.n_envs}"
        logger = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        init_step = 1 if not ckpt.resumed else ckpt.loaded_run['last_step']
        global_step = 0 if not ckpt.resumed else ckpt.loaded_run['global_step']
        start_time = start_time
        obs, _ = envs.reset(seed=args.seed)
       
        try:
            for step in range(init_step, int(args.total_timesteps // args.n_envs)):
                global_step += args.n_envs

                if global_step < args.learning_starts:
                    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                else:
                    with th.no_grad():
                        actions, _, _ = actor.get_action(th.tensor(obs).to(device))
                        actions = actions.detach().cpu().numpy()

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                real_next_obs = next_obs.copy()
                for idx, done in enumerate(np.logical_or(terminations, truncations)):
                    if done: real_next_obs[idx] = infos["final_observation"][idx]
                
                replay_buf.add(obs, real_next_obs, actions, rewards, terminations, infos)

                obs = next_obs

                if global_step % args.eval_freq == 0:
                    evaluator.evaluate(global_step, actor)
                    if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")

                if global_step > args.learning_starts:
                    if global_step % args.train_freq == 0:
                        data = replay_buf.sample(args.batch_size)

                        if continuous_actions:
                            with th.no_grad():
                                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                                critic1_next_tg = tg_critic1(data.next_observations, next_state_actions)
                                critic2_next_tg = tg_critic2(data.next_observations, next_state_actions)
                            
                                min_qf_next_tg = th.min(critic1_next_tg, critic2_next_tg) - alpha * next_state_log_pi
                                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_tg).view(-1)

                            critic1_a_values = critic1(data.observations, data.actions).view(-1)
                            critic2_a_values = critic2(data.observations, data.actions).view(-1)
                            critic1_loss = F.mse_loss(critic1_a_values, next_q_value)
                            critic2_loss = F.mse_loss(critic2_a_values, next_q_value)
                            critic_loss = critic1_loss + critic2_loss

                            # Optimize the model
                            critic_optim.zero_grad()
                            critic_loss.backward()
                            critic_optim.step()

                            if global_step % (args.train_freq * args.actor_train_freq) == 0:  # TD 3 Delayed update support
                                for _ in range(args.actor_train_freq):  # Compensate for the delay by doing 'actor_update_interval' instead of 1
                                    pi, log_pi, _ = actor.get_action(data.observations)
                                    critic1_pi = critic1(data.observations, pi)
                                    critic2_pi = critic2(data.observations, pi)
                                    min_critic_pi = th.min(critic1_pi, critic2_pi)
                                    actor_loss = ((alpha * log_pi) - min_critic_pi).mean()

                                    actor_optim.zero_grad()
                                    actor_loss.backward()
                                    actor_optim.step()

                                    if args.autotune:
                                        with th.no_grad():
                                            _, log_pi, _ = actor.get_action(data.observations)
                                        alpha_loss = (-log_alpha.exp() * (log_pi + tg_entropy)).mean()

                                        alpha_optim.zero_grad()
                                        alpha_loss.backward()
                                        alpha_optim.step()
                                        alpha = log_alpha.exp().item()
                        else: 
                            with th.no_grad():
                                _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                                critic1_next_tg = tg_critic1(data.next_observations)
                                critic2_next_tg = tg_critic2(data.next_observations)
                                # Use the action probabilities instead of MC sampling to estimate the expectation
                                min_qf_next_tg = next_state_action_probs * (
                                    th.min(critic1_next_tg, critic2_next_tg) - alpha * next_state_log_pi
                                )
                                # Adapt Q-target for discrete Q-function
                                min_qf_next_tg = min_qf_next_tg.sum(dim=1)
                                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_tg)

                            # Use Q-values only for the taken actions
                            critic1_a_values = critic1(data.observations).gather(1, data.actions.long()).view(-1)
                            critic2_a_values = critic2(data.observations).gather(1, data.actions.long()).view(-1)
                            
                            critic1_loss = F.mse_loss(critic1_a_values, next_q_value)
                            critic2_loss = F.mse_loss(critic2_a_values, next_q_value)
                            critic_loss = critic1_loss + critic2_loss

                            # Optimize the model
                            critic_optim.zero_grad()
                            critic_loss.backward()
                            critic_optim.step()

                            # Actor training
                            _, log_pi, action_probs = actor.get_action(data.observations)
                            with th.no_grad():
                                critic1_values = critic1(data.observations)
                                critic2_values = critic2(data.observations)
                                min_critic = th.min(critic1_values, critic2_values)
                            # No need for reparameterization, the expectation can be calculated for discrete actions
                            actor_loss = (action_probs * ((alpha * log_pi) - min_critic)).mean()
                            actor_optim.zero_grad()
                            actor_loss.backward()
                            actor_optim.step()

                            if args.autotune:
                                # Re-use action probabilities for temperature loss
                                alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + tg_entropy).detach())).mean()

                                alpha_optim.zero_grad()
                                alpha_loss.backward()
                                alpha_optim.step()
                                alpha = log_alpha.exp().item()

                    # Update the tg networks
                    if global_step % args.tg_critic_freq == 0:
                        for param, tg_param in zip(critic1.parameters(), tg_critic1.parameters()):
                            tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)
                        for param, tg_param in zip(critic2.parameters(), tg_critic2.parameters()):
                            tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)
                
                # If we reach the node's time limit, we just exit the training loop, save metrics and checkpoint
                if (time() - start_time) / 60 >= args.time_limit:
                    break
                
        finally:
             # Save the checkpoint and logger data
            ckpt.set_record(args, alpha, actor, critic1, critic2, global_step, actor_optim, critic_optim, "" if not logger else logger.wb_path, step)
            ckpt.save()
            if logger: logger.close()
            envs.close()
