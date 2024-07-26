import os
import json

from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import grid2op
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace, DiscreteActSpace # if we import gymnasium, GymEnv will convert to Gymnasium!   
from grid2op.Reward import CombinedReward, IncreasingFlatReward, DistanceReward
from lightsim2grid import LightSimBackend

from common.imports import *
from .reward import LineMarginReward, RedispRewardv1
from .heuristic import GridOpRecoAndRevertBus, GridOpIdle

# Get the directory of the current module
ENV_DIR = os.path.dirname(__file__)

def load_config(file_path: str) -> Dict:
    """Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    # Get the directory of the current module (__file__ contains the path of the current file)
    with open(f"{ENV_DIR}/{file_path}", 'r') as file:
        config = json.load(file)
    return config

def norm_action_limits(gym_env: GymEnv, attrs: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Normalize action limits to [0, 1] range for specified attributes.

    Args:
        gym_env: The gym environment.
        attrs: List of action attributes to normalize.

    Returns:
        A tuple containing multiplicative and additive factors for normalization.
    """
    # Getting the right coefficients to have action limits in [0, 1] and use sigmoid output activations
    # (following how grid2op handles mult and add factors: https://github.com/rte-france/Grid2Op/blob/5d938584da3c42dc26fc8128335e20ef382bef00/grid2op/gym_compat/box_gym_actspace.py#L376)
    mult_factor, add_factor = {}, {}
    for attr in attrs:
        attr_box = gym_env.action_space[attr]

        # Filter out elements where both low and high limits are not equal to 0
        feasible_acts = (attr_box.low != attr_box.high)
        low = attr_box.low[feasible_acts]
        high = attr_box.high[feasible_acts]
        
        # Calculate range, multiplicative factor, and additive factor
        range = high - low
        mult_factor[attr] = range
        add_factor[attr] = low

    return mult_factor, add_factor

def make_env(args: Dict[str, Any], idx: int, resume_run: bool = False, generate_class: bool = False, async_vec_env: bool = False) -> Any:
    """Create and configure a grid2op environment.

    Args:
        args: Arguments containing environment configuration parameters.
        idx: Index of the environment instance.
        resume_run: Whether to resume a previous run.
        generate_class: Whether to generate classes for asynchronous environments.
        async_vec_env: Whether the environment is asynchronous.

    Returns:
        A configured grid2op environment wrapped in a GymEnv.
    """
    def thunk():
        config = load_config(args.env_config_path)  # Load environment configuration
        env_id = args.env_id
        env_type = args.action_type.lower()
        difficulty = args.difficulty

        env_config = config['environments']
        assert env_id in env_config.keys(), f"Invalid environment ID: {env_id}. Available IDs are: {env_config.keys()}"

        env_types = ["topology", "redispatch"]
        assert env_type in env_types, f"Invalid environment type: {env_type}. Available IDs are: {env_types}"

        max_difficulty = env_config[env_id]['difficulty']
        assert difficulty < max_difficulty, f"Invalid difficulty: {difficulty}. Difficulty limit is : {max_difficulty-1}"
           
        # Create a grid2op environment with specified backend and reward structure
        g2op_env = grid2op.make(
            env_config[env_id]['grid2op_id'], 
            reward_class=CombinedReward, 
            experimental_read_from_local_dir=True if async_vec_env else False,
            #other_rewards={f"line_{l_id}": N1Reward(l_id=l_id) for l_id in range(186)}
            backend=LightSimBackend()
        ) 
        cr = g2op_env.get_reward_instance()  # Initialize the combined reward instance
        # Per step (cumulative) positive reward for staying alive; reaches 1 at the end of the episode
        cr.addReward("IncreasingFlatReward", 
                    IncreasingFlatReward(per_timestep=1/g2op_env.chronics_handler.max_episode_duration()),
                    0.1)
        if env_type == 'topology': 
           cr.addReward("TopologyReward", DistanceReward(), 0.3)   # = 1 if topology is the original one, 0 if everything changed
        #else:   # TODO remove else -> redispatch should be higher weighted
        cr.addReward("redispatchReward", RedispRewardv1(), 0.3 if env_type == 'topology' else 0.6)  # Custom one, see common.rewards
        cr.addReward("LineMarginReward", LineMarginReward(), 0.3)  # Custom one, see common.rewards
        cr.initialize(g2op_env)  # Finalize the reward setup

        g2op_env.chronics_handler.set_chunk_size(100)    # Instead of loading all episode data, get chunks of 100

        if generate_class:
            g2op_env.generate_classes()
            print("Class generated offline for AsyncVecEnv execution")
            quit()
        
        gym_env = GymEnv(g2op_env, shuffle_chronics=True)  # Wrap the grid2op environment in a GymEnv
        
        # Making sure we can act on 1 sub / line status at the same step
        p = gym_env.init_env.parameters
        p.MAX_LINE_STATUS_CHANGED = 1 
        p.MAX_SUB_CHANGED = 1 
        gym_env.init_env.change_parameters(p)

        # The .reset is required to change the parameters
        if resume_run: gym_env.reset()   # NOTE: to use set_id, we first have to set gym_env's seed with a reset   
        else: gym_env.reset(seed=args.seed+idx)

        gym_env.init_env.chronics_handler.shuffle()  # Shuffle the chronics

        # Prepare action and observation spaces
        state_attrs = config['state_attrs']
        obs_attrs = state_attrs['default']
        if env_config[env_id]['maintenance']: obs_attrs += state_attrs['maintenance']
        #if env_config[env_id]['opponent']: obs_attrs += state_attrs['curtailment']

        if env_type == 'topology': 
            obs_attrs += state_attrs['topology']

            # Set the actions space from the loaded list of (vectorized) actions
            loaded_action_space = np.load(f"{ENV_DIR}/action_spaces/{env_id}_action_space.npy", allow_pickle=True)

            # Increase the action_space size exponentially (from 50) based on difficulty
            n_actions = np.geomspace(50, len(loaded_action_space), num=max_difficulty).astype(int)
            gym_env.action_space = DiscreteActSpace(
                g2op_env.action_space,
                action_list=loaded_action_space[:n_actions[difficulty]]
            )
        else:
            actions_to_keep = ['redispatch', 'curtail']

            obs_attrs += state_attrs['redispatch']
            if env_config[env_id]['renewable']: 
                obs_attrs += state_attrs['curtailment']
            if env_config[env_id]['battery']:
                obs_attrs += state_attrs['storage']
                actions_to_keep += ['set_storage']
        
            # Normalize action limits for selected attributes
            mult_factor, add_factor = norm_action_limits(gym_env, actions_to_keep)

            gym_env.action_space = BoxGymActSpace(
                gym_env.init_env.action_space,
                attr_to_keep=actions_to_keep,
                multiply=mult_factor,   
                add=add_factor
            )
            
        # Set the observation space
        gym_env.observation_space = BoxGymObsSpace(gym_env.init_env.observation_space,
                                            attr_to_keep=obs_attrs,
                                            #divide={"gen_p": gym_env.init_env.gen_pmax,
                                            #        "actual_dispatch": gym_env.init_env.gen_pmax},
        )

        #gym_env = NormalizeReward(gym_env)        
        if args.use_heuristic: 
            if args.heuristic_type == 'idle':
                gym_env = GridOpIdle(gym_env)
            else:
                gym_env = GridOpRecoAndRevertBus(gym_env)
        else: gym_env = gym.wrappers.RecordEpisodeStatistics(gym_env)            
    
        if args.norm_obs: gym_env = NormalizeObservation(gym_env)

        return gym_env
    
    # Return the environment object with custom serialization methods
    return thunk