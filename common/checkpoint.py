import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .imports import *

@dataclass
class CheckpointSaver(ABC):
    """Abstract base class for saving and loading checkpoints.

    Attributes:
        run_name (str): The name of the run.
        args (dict): The arguments for the run configuration.
    """
    run_name: str
    args: dict

    def __post_init__(self):
        """Post-initialization to set up the checkpoint directory and load a checkpoint if resuming a run."""
        self.ckpt_dir = 'checkpoint'
        if not os.path.exists(self.ckpt_dir): os.makedirs(self.ckpt_dir)
        self.loaded_run, self.record = {}, {}
        if self.args.resume_run_name:
            checkpoint_name = 'checkpoint/' + self.args.resume_run_name + '.tar'
            self.loaded_run = th.load(checkpoint_name)
            os.remove(checkpoint_name)
    
    @property
    def resumed(self) -> bool:
        """Check if a run was resumed from a checkpoint.

        Returns:
            True if a run was resumed, False otherwise.
        """
        return self.loaded_run != {}
    
    def _get_base_record(self, global_step: int) -> None:
        """Get a base record with the global step.

        Args:
            global_step: The current global step.
        """
        self.record = {
            'global_step': global_step,
        }

    def save(self) -> None:
        """Save the current record to a checkpoint file."""
        th.save(
            self.record, self.ckpt_dir + '/' + self.run_name + '.tar'
        )

    @abstractmethod
    def set_record(self) -> None:
        """Abstract method to set the record with specific run details."""
        pass
        
class PPOCheckpoint(CheckpointSaver):
    def set_record(
        self, args: Dict[str, Any], actor: nn.Sequential, critic: nn.Sequential, global_step: int, 
        actor_optim: optim, critic_optim: optim, wb_run_name: str, last_rollout: int = 0
    ) -> None:
        """Set the record for PPO checkpoints.

        Args:
            args: Run arguments.
            actor: Actor network.
            critic: Critic network.
            global_step : Current global step.
            actor_optim: Actor optimizer.
            critic_optim: Critic optimizer.
            wb_run_name: Weights & Biases run name.
            last_rollout: Last rollout step. Defaults to 0.
        """
        if global_step >= args.total_timesteps - args.n_envs:
            self.run_name = 'final_' + self.run_name
        self._get_base_record(global_step)
        self.record['args'] = args
        self.record['actor'] = actor.state_dict()
        self.record['critic'] = critic.state_dict()
        self.record['actor_optim'] = actor_optim.state_dict()
        self.record['critic_optim'] = critic_optim.state_dict()
        self.record['wb_run_name'] = wb_run_name
        self.record['last_rollout'] = last_rollout

class LagrPPOCheckpoint(PPOCheckpoint):
    def set_record(
        self, args: Dict[str, Any], actor: nn.Sequential, critic: nn.Sequential, cost_critic: nn.Sequential, global_step: int, actor_optim: optim, critic_optim: optim, cost_critic_optim: optim, cost_threshold, lag_mul, lag_optim: optim, wb_run_name: str, last_rollout: int = 0
    ) -> None:
        """Set the record for PPO checkpoints.

        Args:
            args: Run arguments.
            actor: Actor network.
            critic: Critic network.
            global_step : Current global step.
            actor_optim: Actor optimizer.
            critic_optim: Critic optimizer.
            wb_run_name: Weights & Biases run name.
            last_rollout: Last rollout step. Defaults to 0.
        """
        super().set_record(args, actor, critic, global_step, actor_optim, critic_optim, wb_run_name, last_rollout)
        self.record['cost_critic'] = cost_critic.state_dict()
        self.record['cost_critic_optim'] = cost_critic_optim.state_dict()
        self.record['cost_threshold'] = cost_threshold
        self.record['lag_mul'] = lag_mul.tolist()
        self.record['lag_optim'] = lag_optim.state_dict()

class SACCheckpoint(PPOCheckpoint):
    def set_record(
        self, args: Dict[str, Any], alpha: float, actor: nn.Sequential, critic: nn.Sequential, critic2: nn.Sequential, global_step: int, actor_optim: optim, critic_optim: optim, wb_run_name: str, last_step: int = 0
    ) -> None:
        """Set the record for SAC checkpoints.

        Args:
            args: Run arguments.
            alpha: Alpha parameter.
            actor: Actor network.
            critic: Critic network.
            critic2: Second critic network.
            global_step: Current global step.
            actor_optim: Actor optimizer.
            critic_optim: Critic optimizer.
            wb_run_name: Weights & Biases run name.
            last_step: Last step. Defaults to 0.
        """
        super().set_record(args, actor, critic, global_step, actor_optim, critic_optim, wb_run_name)
        self.record['critic2'] = critic2.state_dict()
        self.record['alpha'] = alpha
        self.record['last_step'] = last_step

class TD3Checkpoint(PPOCheckpoint):
    def set_record(
        self, args: Dict[str, Any], actor: nn.Sequential, critic: nn.Sequential, critic2: nn.Sequential, global_step: int, actor_optim: optim, critic_optim: optim, wb_run_name: str, last_step: int = 0
    ) -> None:
        """Set the record for TD3 checkpoints.

        Args:
            args: Run arguments.
            actor: Actor network.
            critic: Critic network.
            critic2: Second critic network.
            global_step: Current global step.
            actor_optim: Actor optimizer.
            critic_optim: Critic optimizer.
            wb_run_name: Weights & Biases run name.
            last_step: Last step. Defaults to 0.
        """
        super().set_record(args, actor, critic, global_step, actor_optim, critic_optim, wb_run_name)
        self.record['critic2'] = critic2.state_dict()
        self.record['last_step'] = last_step

class DQNCheckpoint(CheckpointSaver):
    def set_record(
        self, args: Dict[str, Any], qnet: nn.Sequential, global_step: int, qnet_optim: optim, wb_run_name: str, last_step: int = 0
    ) -> None:
        """Set the record for DQN checkpoints.

        Args:
            args: Run arguments.
            qnet: Q-network.
            global_step: Current global step.
            qnet_optim: Q-network optimizer.
            wb_run_name: Weights & Biases run name.
            last_step: Last step. Defaults to 0.
        """
        if global_step >= args.total_timesteps - args.n_envs:
            self.run_name = 'final_' + self.run_name
        self._get_base_record(global_step)
        self.record['args'] = args
        self.record['qnet'] = qnet.state_dict()
        self.record['qnet_optim'] = qnet_optim.state_dict()
        self.record['wb_run_name'] = wb_run_name
        self.record['last_step'] = last_step
