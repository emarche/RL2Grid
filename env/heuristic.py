from abc import ABC, abstractmethod
from common.imports import *

RHO_SAFETY_THRESHOLD = 0.95

class GridOp(gym.Wrapper, ABC):
    """Abstract base class for heuristic operation wrappers.

    This class provides methods to apply heuristic actions to a grid environment
    to maintain stability and avoid risky situations.

    Attributes:
        init_env (GymEnv): The initial environment to wrap.
        ep_reward (float): Cumulative reward for the current episode.
    """
    def __init__(self, gym_env, eval_env: bool = False):
        """
        Initialize the GridOp class.

        Args:
            gym_env (gym.Env): The environment to wrap.
            custom_param (optional): Any custom parameter for the GridOp wrapper.
        """
        super().__init__(gym_env)  # Initialize the gym.Wrapper part
        self.eval_env = eval_env
        self.n_rewards = 1

    def set_n_rewards(self, n_rewards: int):
        """
        Sets the number of rewards to track for the evaluation environment.

        Args:
            n_rewards (int): The number of rewards to set.
        """
        self.n_rewards = n_rewards

    @property
    def _risk_overflow(self) -> bool:
        """Check if the maximum rho value exceeds the safety threshold."""
        return self.init_env.current_obs.rho.max() >= RHO_SAFETY_THRESHOLD

    @property  
    def _obs(self) -> np.ndarray:
        """Get the current observation from the Grid2Op environment."""
        return self.init_env.current_obs
    
    @abstractmethod
    def _get_heuristic_actions(self) -> List:
        """Get a list of heuristic actions to apply to the environment."""
        return []

    def apply_actions(self) -> Tuple[float, bool, Dict]:
        """Apply heuristic actions until a risky situation or episode end.

        Returns:
            A tuple containing the cumulative heuristic reward, a boolean indicating
            if the episode is done, and additional info.
        """
        use_heuristic = True
        done, info = False, {}
        while use_heuristic:
            g2o_actions = self._get_heuristic_actions()
            if not g2o_actions: break
            for g2o_act in g2o_actions:
                _, reward, done, info = self.init_env.step(g2o_act)
                self.ep_reward += reward    # Cumulate episode reward over heuristic steps
                if self.eval_env: self.ep_rewards += list(info['rewards'].values())

                if done or self._risk_overflow:   # Resume the agent if in a risky situation
                    use_heuristic = False
                    break

        return done, info

    def step(self, gym_action: Union[int, List[float]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment using the given gym action, then follow the heuristic.

        Args:
            gym_action: The action to take in the gym environment.

        Returns:
            A tuple containing the observation, reward, done flag, truncation flag, and additional info.
        """
        _, reward, done, info = self.init_env.step(self.action_space.from_gym(gym_action))
        self.ep_reward += reward
        if self.eval_env: self.ep_rewards += list(info['rewards'].values())

        if not done and not self._risk_overflow:
            done, info = self.apply_actions()

        if done: info['episode'] = {'l': [self.init_env.nb_time_step], 'r': [self.ep_reward]}    # Replace the use of RecordEpisodeStatistics

        if self.eval_env: info['rewards'] = dict(zip(info['rewards'].keys(), self.ep_rewards))

        return (
            self.observation_space.to_gym(self._obs), 
            float(reward), 
            done, 
            False,  # Truncation is typically False in g2o envs
            info    
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        Args:
            **kwargs: Additional arguments to pass to the reset method.

        Returns:
            A tuple containing the initial observation and additional info.
        """
        done = True     # It could happen that an episode ends in the reset step
        while done:     # Without this it would not reset and crash on the env.step(...)
            _, info = super().reset(**kwargs)  # Reset the underlying scenario
            self.ep_reward = 0.
            self.ep_rewards = np.zeros(self.n_rewards)
            if not self._risk_overflow: done, info = self.apply_actions()

        if self.eval_env: info['rewards'] = dict(zip(info['rewards'].keys(), self.ep_rewards))

        return self.observation_space.to_gym(self._obs), info

class GridOpIdle(GridOp):
    """A grid operation wrapper that remains idle unless in a risky situation.

    This class defines heuristic actions that take no action if the environment
    is in a safe state and only acts when the safety threshold is exceeded.
    """

    def _get_heuristic_actions(self) -> List:
        """Return an empty list of actions if risk overflow, otherwise return a default action."""   
        if self._risk_overflow: return []
        else: return [self.init_env.action_space()]    
        
class GridOpReco(GridOp):
    """A grid operation wrapper that focuses on reconnecting lines.

    This class defines heuristic actions to reconnect power lines that are
    out of cooldown and restores the grid's connectivity.
    """

    def _get_line_reconnect_actions(self) -> List:
        """Get actions to reconnect lines that are out of cooldown."""
        to_reconnect = (~self._obs.line_status) & (self._obs.time_before_cooldown_line == 0)
        if np.any(to_reconnect):
            reco_id = np.where(to_reconnect)[0]     # Reconnect lines that are out of cooldown
            return [self.init_env.action_space({"set_line_status": [(line_id, 1)]}) for line_id in reco_id]
        return []
    
    def _get_heuristic_actions(self) -> List:
        """Get heuristic actions to reconnect lines or a default action."""
        if self._risk_overflow: return []
        actions = self._get_line_reconnect_actions()
        if np.any(actions): return actions
        return [self.init_env.action_space()]    

class GridOpRevertBus(GridOp):
    """A grid operation wrapper that focuses on reverting bus changes.

    This class defines heuristic actions to revert bus changes at substations
    that have had bus changes and are out of cooldown.
    """

    def _get_bus_revert_actions(self) -> List:
        """Get actions to revert bus changes at substations."""        
        # For each substation, count the objects that have changed bus 
        bus_changed_info = np.array([
            np.count_nonzero(sub_buses > 1) 
             for sub_buses in np.split(self._obs.topo_vect, np.cumsum(self._obs.sub_info[:-1]))
        ])

        # Get the idx of most changed substation that is out of cooldown
        # Get the subs out of cooldown where there have been some bus changes
        revertable_subs = bus_changed_info[(self._obs.time_before_cooldown_sub == 0) & (bus_changed_info > 0)]      

        if revertable_subs.size > 0:    # if there is a sub to revert           
            # Get unique values in revertable_subs and their counts
            unique_subs, counts = np.unique(revertable_subs, return_counts=True)

            # Create a dictionary to map those unique values to their inverted counts (descending) - handles duplicates
            ordered_sub_values = dict(zip(unique_subs, np.argsort(counts)[::-1]))

            # Find indexes where elements in bus_changed_info match values in revertable_subs
            bus_changed_idxs = np.where(np.in1d(bus_changed_info, unique_subs))[0]

            # Get the corresponding order for each matching element using the dictionary
            bus_changed_orders = [ordered_sub_values[sub] for sub in bus_changed_info[bus_changed_idxs]]
            # Sort and rank indexes based on the orders (descending order)
            bus_changed_idxs = bus_changed_idxs[np.argsort(bus_changed_orders)]
            
            return [self.init_env.action_space(
                {"set_bus": 
                 {"substations_id": [(revert_sub_idx, np.ones(self._obs.sub_info[revert_sub_idx], dtype=int))]}})
                for revert_sub_idx in bus_changed_idxs
            ]
        return []
    
    def _get_heuristic_actions(self) -> List:
        """Get heuristic actions to revert bus changes or a default action."""        
        if self._risk_overflow: return []
        actions = self._get_bus_revert_actions()
        if np.any(actions): return actions
        return [self.init_env.action_space()]    

class GridOpRecoAndRevertBus(GridOpReco, GridOpRevertBus):   
    """A grid operation wrapper that combines reconnecting lines and reverting bus changes.

    This class defines heuristic actions to both reconnect power lines that are
    out of cooldown and revert bus changes at substations that have had bus changes
    and are out of cooldown.
    """

    def _get_heuristic_actions(self) -> List:
        """Get heuristic actions to reconnect lines and revert bus changes."""
        if self._risk_overflow: return []

        actions = self._get_line_reconnect_actions()
        actions.extend(self._get_bus_revert_actions())
      
        if actions: return actions
        return [self.init_env.action_space()]    

'''
# Some useful Grid2Op shortcuts

# Get the substations impacted by an action (as a True/False mask with n_sub dimension)
#   env.action_space.from_gym(3)._subs_impacted

# Get the idx of an object in the topo_vect
#   obs.line_or_pos_topo_vect
#   obs.line_ex_pos_topo_vect
#   obs.load_pos_topo_vect
#   obs.gen_pos_topo_vect
#   obs.storage_pos_topo_vect

# Get the bus on which the objects are connected
#   obs.topo_vect[obs.line_or_pos_topo_vect]

'''