from common.imports import *
from common.utils import Linear, th_act_fns

class QNetwork(nn.Module):
    """Q-Network for a reinforcement learning agent.

    This network predicts Q-values for given states, allowing the agent to select actions.

    Attributes:
        qnet (nn.Sequential): Sequential neural network model for Q-value prediction.
    """

    def __init__(self, envs: gym.Env, args: Dict[str, Any]):
        """Initialize the Q-Network.

        Args:
            envs: Environment(s) with defined observation and action spaces.
            args: Arguments containing network configuration, including activation function and layer sizes.
        """
        super().__init__()

        act_str, act_fn = args.act_fn, th_act_fns[args.act_fn]

        layers = []
        layers.extend([
            Linear(np.array(envs.single_observation_space.shape).prod(), args.layers[0], act_str), 
            act_fn
        ])
        for idx, embed_dim in enumerate(args.layers[1:], start=1): 
            layers.extend([Linear(args.layers[idx-1], embed_dim, act_str), act_fn])
        
        layers.append(Linear(args.layers[-1], envs.single_action_space.n, 'linear'))

        self.qnet = nn.Sequential(*layers)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass through the Q-Network.

        Args:
            x: Input tensor representing the state.

        Returns:
            Tensor with Q-values for each action.
        """
        return self.qnet(x)

    def get_action(self, x: th.Tensor) -> np.ndarray:
        """Get the action with the highest Q-value.

        Args:
            x: Input tensor representing the state.

        Returns:
            Numpy array of selected actions.
        """
        q_values = self(x)
        actions = th.argmax(q_values, dim=1)
        return actions
    
    def get_eval_action(self, x: th.Tensor) -> np.ndarray:
        """Get the action for evaluation.

        Args:
            x: Input tensor representing the state.

        Returns:
            Numpy array of selected actions for evaluation.
        """
        q_values = self(x)
        actions = th.argmax(q_values)
        return actions
