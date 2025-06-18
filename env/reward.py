from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward import RedispReward, IncreasingFlatReward, FlatReward, DistanceReward
from grid2op.dtypes import dt_float

from common.imports import *
from common.logger import Logger

import grid2op

from lightsim2grid import ContingencyAnalysis
from lightsim2grid.rewards import N1ContingencyReward

class DistanceRewardv1(DistanceReward):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(0.0)

class IncreasingFlatRewardv1(IncreasingFlatReward):
    def __init__(self, per_timestep=1, logger=None):
        super().__init__(per_timestep, logger)
        self.reward_min = dt_float(-10.0)

class FlatRewardv1(FlatReward):
    def __init__(self, per_timestep=1, logger=None):
        super().__init__(per_timestep, logger)
        self.reward_min = dt_float(-5.0)

class OverloadReward(BaseReward):
    def __init__(self, logger: Optional[Logger] = None, constrained: bool = False):
        super().__init__(logger=logger)
        self.penalty = dt_float(-1.0 if not constrained else 0.0)
        self.min_reward = dt_float(-5.0)

    def __call__(self, action: np.ndarray, env: gym.Env, has_error: bool, is_done: bool, is_illegal: bool, is_ambiguous: bool) -> float:
        if has_error or is_illegal or is_ambiguous:  
            return self.min_reward
        
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        margin = np.divide(thermal_limits - ampere_flows, thermal_limits + 1e-10)
        penalty_disconnection = self.penalty * sum(~env.current_obs.line_status) / (env.current_obs.n_line * 0.1)
        penalty_overload = margin[margin < 0].sum() / (env.current_obs.n_line * 0.1)

        return penalty_overload + penalty_disconnection

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
        self.min_reward = dt_float(-1.0)
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
        if has_error or is_illegal or is_ambiguous:      
            return self.min_reward
        
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        margin = np.divide(thermal_limits - ampere_flows, thermal_limits + 1e-10)

        # Reward is based on how much lines are used (the lower the better and goes negative in case of overflow) and is penalized for disconnected lines. We then normalize everything between ~[-1, 1]  
        reward = margin[env.current_obs.line_status].sum() + (self.penalty * sum(~env.current_obs.line_status) / env.current_obs.n_line)

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

        if has_error or is_illegal or is_ambiguous:      
            return -1.      # self.reward_min

        # Compute the losses
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        # Don't forget to convert MW to MWh !
        losses = (gen_p.sum() - load_p.sum()) * env.delta_time_seconds / 3600.0
        # Compute the marginal cost
        marginal_cost = np.max(env.gen_cost_per_MW[env._gen_activeprod_t > 0.0])
        # Redispatching amount is env._actual_dispatch
        redisp_cost = (
            self._alpha_redisp * np.abs(env._actual_dispatch).sum() * marginal_cost * env.delta_time_seconds / 3600.0
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


class N1ContingencyRewardv1(N1ContingencyReward):         
    def initialize(self, env: "grid2op.Environment.Environment"):
        super().initialize(env)        
        self.reward_min = -self.reward_max

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        
        '''
        # See the LineMarginReward for an explanation 
        if is_done:
            if has_error or is_illegal or is_ambiguous:
                return self.reward_min
        '''

        # retrieve the state of the grid
        self._backend_action.reset()
        act = env.backend.get_action_to_set()
        th_lim_a = 1. * env.get_thermal_limit()
        th_lim_a[th_lim_a <= 1.] = 1.  # assign 1 for the thermal limit
        
        # apply it to the backend
        self._backend_action += act
        self._backend.apply_action(self._backend_action)
        conv, exc_ = self._backend.runpf()
        if not conv:
            self.logger.warn("Cannot set the backend of the `N1ContingencyReward` => divergence")
            return self.reward_min
        
        # synch the contingency analyzer
        contingecy_analyzer = ContingencyAnalysis(self._backend)
        contingecy_analyzer.computer.change_solver(self._solver_type)
        contingecy_analyzer.add_multiple_contingencies(*self._l_ids)
        tmp = contingecy_analyzer.get_flows()
        self.logger.info(f"{contingecy_analyzer.computer.nb_solved()} converging contingencies")
        if self._dc:
            # In DC is study p, but take into account q in the limits
            tmp_res = np.abs(tmp[0])  # this is Por
            # now transform the limits in A in MW
            por, qor, vor, aor = env.backend.lines_or_info()
            p_sq = (1e-3 * th_lim_a)**2 * 3. * vor**2 - qor**2
            p_sq[p_sq <= 0.] = 0.
            limits = np.sqrt(p_sq)
        else:
            tmp_res = 1. * tmp[1]
            limits = th_lim_a
        res = ((tmp_res > self._threshold_margin * limits) | (~np.isfinite(tmp_res))).any(axis=1)  # whether one powerline is above its limit, per cont
        res |=  (np.abs(tmp_res) <= self._tol).all(axis=1)  # other type of divergence: all 0.
        res = res.sum()  # count total of n-1 unsafe 
        if self._normalize: res /= len(self._l_ids)
        return -res     # emarche: we want penalize the agent based on N1 contingencies
    
    def close(self):
        if self._backend is not None:
            self._backend.close()
        del self._backend
        self._backend = None