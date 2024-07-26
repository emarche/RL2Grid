from torch.distributions import Normal, Categorical

from common.imports import *
from common.utils import Linear, th_act_fns

class QNetwork(nn.Module):
    """Critic network (Q-network) for the TD3 algorithm.

    Attributes:
        fc_i (Linear): Input layer of the critic network.
        fc_h (List[nn.Linear]): A list of hidden layers
        fc_o (Linear): Output layer of the critic network.
        act_fn (function): Activation function used in the network.
    """

    def __init__(self, envs: gym.Env, args: Dict[str, Any]):
        super().__init__()

        critic_layers = args.critic_layers
        act_str, self.act_fn = args.critic_act_fn, th_act_fns[args.critic_act_fn]

        self.fc_i = Linear(
            np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape),      
            critic_layers[0], 
            act_str
        )
        
        self.fc_h = []
        for idx, embed_dim in enumerate(critic_layers[1:], start=1): 
            self.fc_h.append(Linear(critic_layers[idx-1], embed_dim, act_str))

        else: self.fc_o = Linear(critic_layers[-1], 1, 'linear')

    def forward(self, x: th.Tensor, a: th.Tensor) -> th.Tensor:
        """Forward pass of the Q-network.

        Args:
            x: Input observations.
            a: Action input tensor.

        Returns:
            A tensor containing the Q-value.
        """
        x = self.act_fn(self.fc_i(th.cat([x, a], 1)))
        for fc in self.fc_h: x = self.act_fn(fc(x))
        return self.fc_o(x)

class Actor(nn.Module):
    """Actor network for the TD3 algorithm.

    Attributes:
        fc_i (nn.Linear): Input layer of the actor network.
        fc_h (List[nn.Linear]): A list of hidden layers
        fc_mu (nn.Linear): Output layer providing mean of the action distribution.
        action_scale (th.Tensor): Scaling factor for action output.
        action_bias (th.Tensor): Bias factor for action output.
        act_fn (nn): Activation function used in the network.
    """
    def __init__(self, envs: gym.Env, args: Dict[str, Any]):
        super().__init__()
       
        actor_layers = args.actor_layers
        act_str, self.act_fn = args.actor_act_fn, th_act_fns[args.actor_act_fn]
        self.fc_i = Linear(
            np.prod(envs.single_observation_space.shape), actor_layers[0], 
            act_str
        )
       
        self.fc_h = []
        for idx, embed_dim in enumerate(actor_layers[1:], start=1): 
            self.fc_h.append(Linear(actor_layers[idx-1], embed_dim, act_str))

        self.fc_mu = Linear(actor_layers[idx], np.prod(envs.single_action_space.shape), 'tanh')
        # action rescaling
        self.register_buffer(
            "action_scale", th.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0, dtype=th.float32)
        )
        self.register_buffer(
            "action_bias", th.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0, dtype=th.float32)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass of the actor network.

        Args:
            x: Input observations.

        Returns:
            Scaled action tensor.
        """
        x = self.act_fn(self.fc_i(x))
        for fc in self.fc_h: x = self.act_fn(fc(x))
        x = th.tanh_(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
    
    def get_eval_action(self, x: th.Tensor) -> th.Tensor:
        """
        Get deterministic action for evaluation (without exploration noise).

        Args:
            x: Input observations.

        Returns:
            A tensor with deterministic continuous actions for evaluation.
        """
        return self(x)