from common.imports import *

class ConstrainedFailureGridOp(gym.Wrapper):
    def step(self, gym_action: Union[int, List[float]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment using the given gym action, then return a cost function in the info.

        Args:
            gym_action: The action to take in the gym environment.

        Returns:
            A tuple containing the observation, reward, done flag, truncation flag, and additional info (with the cost).
        """
        next_obs, reward, terminated, truncated, info = super().step(gym_action)
        
        if np.logical_or(terminated, truncated): info['cost'] = 1
        else: info['cost'] = 0
        
        return (
            next_obs, 
            reward, 
            terminated, 
            truncated,  # Truncation is typically False in g2o envs
            info    
        )

class ConstrainedOverloadGridOp(gym.Wrapper):
    def step(self, gym_action: Union[int, List[float]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment using the given gym action, then return a cost function in the info.

        Args:
            gym_action: The action to take in the gym environment.

        Returns:
            A tuple containing the observation, reward, done flag, truncation flag, and additional info (with the cost).
        """
        next_obs, reward, terminated, truncated, info = super().step(gym_action)

        n_disconnections = sum(~self.init_env.current_obs.line_status)
        n_overloads = 0     # If it's game over and the grid is disconnected (i..e, n_disconnections = n_lines), then we cannot get the thermal limit and we don't have to compute n_overloads

        if not np.logical_or(terminated, truncated):
            ampere_flows = np.abs(self.init_env.backend.get_line_flow())
            thermal_limits = np.abs(self.init_env.get_thermal_limit())        
            margin = thermal_limits - ampere_flows
            n_overloads = len(margin[margin < 0])
        
        info['cost'] = n_disconnections + n_overloads

        return (
            next_obs, 
            reward, 
            terminated, 
            truncated,  # Truncation is typically False in g2o envs
            info    
        )
