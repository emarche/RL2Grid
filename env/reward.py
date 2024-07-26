from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward import RedispReward
from grid2op.dtypes import dt_float

from common.imports import *
from common.logger import Logger

class LineMarginReward(BaseReward):
    """A reward function that penalizes disconnections and rewards lower usage of power lines.

    Attributes:
        penalty (dt_float): Penalty value for disconnected lines.
    """
    def __init__(self, logger: Optional[Logger] = None):
        """Initialize the LineMarginReward.

        Args:
            logger: Logger instance for logging information. Defaults to None.
        """
        super().__init__(logger=logger)
        self.penalty = dt_float(-1.0)

    def __call__(self, action: np.ndarray, env: gym.Env, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        """Calculate the reward for the given state of the environment.

        Args:
            action: The action taken.
            env: The environment instance.
            has_error: Whether there was an error.
            is_done: Whether the episode is done.
            is_illegal: Whether the action was illegal.
            is_ambiguous: Whether the action was ambiguous.

        Returns:
            The float calculated reward.
        """
        if has_error:
            return self.penalty * sum(~env.current_obs.line_status) / env.current_obs.n_line
        
        if is_illegal or is_ambiguous:
            return 0.0

        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        margin = np.divide(thermal_limits - ampere_flows, thermal_limits + 1e-10)

        # Reward is based on how much lines are used (the lower the better and goes negative in case of overflow) and is penalized for disconnected lines. We then normalize everything between (more or less) [-1, 1]  
        reward = margin[env.current_obs.line_status].sum() + (self.penalty * sum(~env.current_obs.line_status))

        return reward / env.current_obs.n_line
      
class RedispRewardv1(RedispReward):
    """A reward function that penalizes redispatching costs, losses, and storage usage.

    Inherits from RedispReward.
    """
    def __call__(self, action: np.ndarray, env: gym.Env, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        """Calculate the reward for the given state of the environment.

        Args:
            action: The action taken.
            env: The environment instance.
            has_error: Whether there was an error.
            is_done: Whether the episode is done.
            is_illegal: Whether the action was illegal.
            is_ambiguous: Whether the action was ambiguous.

        Returns:
            The float calculated reward.
        """
        if is_done:
            # If the episode is over (blackout) and it's my fault 
            if has_error or is_illegal or is_ambiguous:
                return self.reward_min
        elif is_illegal or is_ambiguous:
            return self._reward_illegal_ambiguous

        # Compute the losses
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        # Don't forget to convert MW to MWh !
        losses = (gen_p.sum() - load_p.sum()) * env.delta_time_seconds / 3600.0
        # Compute the marginal cost
        gen_activeprod_t = env._gen_activeprod_t
        marginal_cost = np.max(env.gen_cost_per_MW[gen_activeprod_t > 0.0])
        # Redispatching amount
        actual_dispatch = env._actual_dispatch
        redisp_cost = (
            self._alpha_redisp * np.abs(actual_dispatch).sum() * marginal_cost * env.delta_time_seconds / 3600.0
        )
        
        # Cost of losses
        losses_cost = losses * marginal_cost

        # Cost of storage
        c_storage = np.abs(env._storage_power).sum() * marginal_cost * env.delta_time_seconds / 3600.0
        
        # Total "regret"
        regret = losses_cost + redisp_cost + c_storage

        # Compute reward and normalize
        reward = dt_float(-regret/self.max_regret)

        return reward


