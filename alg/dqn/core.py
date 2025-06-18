from time import time

from stable_baselines3.common.buffers import ReplayBuffer

from .agent import QNetwork
from .config import get_alg_args
from common.checkpoint import CheckpointSaver
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator

def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Calculate the linear schedule for epsilon decay.

    Args:
        start_e: Starting epsilon value.
        end_e : Ending epsilon value.
        duration: Total duration over which epsilon decays.
        t: Current timestep.

    Returns:
        The current epsilon value based on the linear decay.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class DQN:        
    """Deep Q-Network (DQN) implementation for training an agent in a given environment: https://arxiv.org/abs/1312.5602.
    """

    def __init__(self, envs: gym.Env, run_name: str, start_time: float, args: Dict[str, Any], ckpt: CheckpointSaver):     
        """Init method for DQN

        Args:
            envs (gym.Env): The environments used for training.
            run_name (str): The name of the current training run.
            start_time (float): The time when training started.
            args (Dict[str, Any]): The command line arguments for configuration.
            ckpt (CheckpointSaver): The checkpoint handler for saving and loading training state.
        """
        # Load algorithm-specific arguments if not resuming from a checkpoint
        if not ckpt.resumed: args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        assert args.train_freq % args.n_envs == 0, \
            f"Invalid train frequency: {args.train_freq}. Must be multiple of n_envs {args.n_envs}"

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

        # Initialize the Q-networks, optimizer, and buffer
        qnet = QNetwork(envs, args).to(device)
        if ckpt.resumed: qnet.load_state_dict(ckpt.loaded_run['qnet'])

        qnet_optim = optim.Adam(qnet.parameters(), lr=args.lr)
        if ckpt.resumed: qnet_optim.load_state_dict(ckpt.loaded_run['qnet_optim'])
        
        tg_qnet = QNetwork(envs, args).to(device)
        tg_qnet.load_state_dict(qnet.state_dict())

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
            for step in range(init_step, int(args.total_timesteps // args.n_envs)):
                global_step += args.n_envs
                epsilon = linear_schedule(
                    args.eps_start, args.eps_end, args.eps_decay_frac * args.total_timesteps, global_step
                )
                if np.random.rand() < epsilon:
                    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                else:
                    with th.no_grad():
                        actions = qnet.get_action(th.tensor(obs).to(device)).cpu().numpy()

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                real_next_obs = next_obs.copy()
                for idx, done in enumerate(np.logical_or(terminations, truncations)):
                    if done: real_next_obs[idx] = infos["final_observation"][idx]
                
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

                obs = next_obs

                if global_step % args.eval_freq == 0:
                    evaluator.evaluate(global_step, qnet)
                    if args.verbose: print(f"SPS={int(global_step / (time() - start_time))}")

                if global_step > args.learning_starts:
                    if global_step % args.train_freq == 0:
                        data = rb.sample(args.batch_size)
                        with th.no_grad():
                            tg_act = qnet(data.next_observations).argmax(dim=1, keepdim=True)
                            tg_mac = tg_qnet(data.next_observations).gather(1, tg_act).squeeze()
                            td_target = data.rewards.flatten() + args.gamma * tg_mac * (1 - data.dones.flatten())

                        old_val = qnet(data.observations).gather(1, data.actions).squeeze()
                        
                        loss = F.mse_loss(td_target, old_val)
                        
                        # Optimize the model
                        qnet_optim.zero_grad()
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(qnet.parameters(), 10)
                        qnet_optim.step()
                        
                    # Update target network
                    if global_step % args.tg_qnet_freq == 0:
                        for tg_qnet_param, qnet_param in zip(tg_qnet.parameters(), qnet.parameters()):
                            tg_qnet_param.data.copy_(
                                args.tau * qnet_param.data + (1.0 - args.tau) * tg_qnet_param.data
                            )

                # If we reach the node's time limit, we just exit the training loop, save metrics and checkpoint
                if (time() - start_time) / 60 >= args.time_limit:
                    break

        finally:
            # Save the checkpoint and logger data
            ckpt.set_record(args, qnet, global_step, qnet_optim, "" if not logger else logger.wb_path, step)
            ckpt.save()
            if logger: logger.close()
            envs.close()
