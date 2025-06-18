from time import time

from stable_baselines3.common.buffers import ReplayBuffer

from .agent import Actor, QNetwork
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator

class TD3:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm implementation: https://arxiv.org/abs/1802.09477.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):
        """Init method for TD3
        
        Attributes:
            envs (gym.Env): Environment object providing observation and action spaces.
            run_name (str): Name of the training run.
            start_time (float): Start time of the training.
            args (Dict[str, Any]): Arguments containing hyperparameters.
            ckpt (CheckpointSaver): Checkpoint object for saving/loading training state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.train_freq % args.n_envs == 0, \
            f"Invalid train frequency: {args.train_freq}. Must be multiple of n_envs {args.n_envs}"

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # Initialize the actor and critic networks, optimizer, and buffer
        actor = Actor(envs, args).to(device)
        critic1 = QNetwork(envs, args).to(device)
        critic2 = QNetwork(envs, args).to(device)

        if ckpt.resumed:
            actor.load_state_dict(ckpt.loaded_run['actor'])
            critic1.load_state_dict(ckpt.loaded_run['critic'])
            critic2.load_state_dict(ckpt.loaded_run['critic2'])
 
        tg_actor = Actor(envs, args).to(device)
        tg_actor.load_state_dict(actor.state_dict())
        tg_critic1 = QNetwork(envs, args).to(device)
        tg_critic2 = QNetwork(envs, args).to(device)
        tg_critic1.load_state_dict(critic1.state_dict())
        tg_critic2.load_state_dict(critic2.state_dict())

        actor_optim = optim.Adam(list(actor.parameters()), lr=args.actor_lr)
        critic_optim = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=args.critic_lr)
        if ckpt.resumed:
            actor_optim.load_state_dict(ckpt.loaded_run['actor_optim'])
            critic_optim.load_state_dict(ckpt.loaded_run['critic_optim'])

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
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
            for step in range(init_step, args.total_timesteps):
                global_step += args.n_envs

                if global_step < args.learning_starts:
                    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                else:
                    with th.no_grad():
                        actions = actor(th.tensor(obs).to(device))
                        actions += th.normal(0, actor.action_scale * args.exploration_noise)
                        actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                real_next_obs = next_obs.copy()
                for idx, done in enumerate(np.logical_or(terminations, truncations)):
                    if done: real_next_obs[idx] = infos["final_observation"][idx]
                
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

                obs = next_obs

                if global_step % args.eval_freq == 0:
                    evaluator.evaluate(global_step, actor)
                    if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")

                if global_step > args.learning_starts:
                    if global_step % args.train_freq == 0:
                        data = rb.sample(args.batch_size)
                        with th.no_grad():
                            clipped_noise = (th.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                                -args.noise_clip, args.noise_clip
                            ) * tg_actor.action_scale

                            next_state_actions = (tg_actor(data.next_observations) + clipped_noise).clamp(
                                envs.single_action_space.low[0], envs.single_action_space.high[0]
                            )
                            critic1_next_tg = tg_critic1(data.next_observations, next_state_actions)
                            critic2_next_tg = tg_critic2(data.next_observations, next_state_actions)
                            min_qf_next_tg = th.min(critic1_next_tg, critic2_next_tg)
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
                            actor_loss = -critic1(data.observations, actor(data.observations)).mean()
                            actor_optim.zero_grad()
                            actor_loss.backward()
                            actor_optim.step()

                    if global_step % args.tg_freq == 0:
                        # Update the tg network
                        for param, tg_param in zip(actor.parameters(), tg_actor.parameters()):
                            tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)
                        for param, tg_param in zip(critic1.parameters(), tg_critic1.parameters()):
                            tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)
                        for param, tg_param in zip(critic2.parameters(), tg_critic2.parameters()):
                            tg_param.data.copy_(args.tau * param.data + (1 - args.tau) * tg_param.data)

                # If we reach the node's time limit, we just exit the training loop, save metrics and checkpoint
                if (time() - start_time) / 60 >= args.time_limit:
                    break
        
        finally:
            # Save the checkpoint and logger data
            ckpt.set_record(args, actor, critic1, critic2, global_step, actor_optim, critic_optim, "" if not logger else logger.wb_path, step)
            ckpt.save()
            if logger: logger.close()
            envs.close()
