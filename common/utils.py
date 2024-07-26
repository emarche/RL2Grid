from argparse import ArgumentTypeError

from torch.utils.tensorboard import SummaryWriter

from .imports import *

seed = 0    # Global variable to store the seed

def set_torch(n_threads: int = 0, deterministic: bool = True, cuda: bool = False) -> th.device:
    """Configure PyTorch settings including the number of threads, determinism, and CUDA usage.

    Args:
        n_threads: Number of threads for PyTorch operations.
        deterministic: Whether to use deterministic algorithms.
        cuda: Whether to enable CUDA.

    Returns:
        The PyTorch device configured based on availability and input flags.
    """
    th.set_num_threads(n_threads)
    th.backends.cudnn.deterministic = deterministic
    return th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

def set_random_seed(s: Optional[int] = None) -> None:
    """Set the random seed for reproducibility across different libraries.

    Args:
        s: Seed value to set. If None, the global seed value is used.
    """
    global seed
    if s is not None: seed = s
    rnd.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def str2bool(s: str) -> bool:
    """Convert a string representation of a boolean to an actual boolean value.

    Args:
        s: String to convert.

    Returns:
        The boolean value corresponding to the input string.

    Raises:
        ArgumentTypeError: If the string does not represent a boolean value.
    """
    if s.lower() == 'true':return True
    elif s.lower() == 'false': return False
    raise ArgumentTypeError('Boolean value expected.')


def Linear(input_dim: int, output_dim: int, act_fn: str = 'leaky_relu', init_weight_uniform: bool = True) -> nn.Linear:
    """Create and initialize a linear layer with appropriate weights.
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/    
    
    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        act_fn: Activation function.
        init_weight_uniform: Whether to uniformly sample initial weights.

    Returns:
        The initialized layer.
    """
    act_fn = act_fn.lower()
    gain = nn.init.calculate_gain(act_fn)
    layer = nn.Linear(input_dim, output_dim)
    if init_weight_uniform: nn.init.xavier_uniform_(layer.weight, gain=gain)
    else: nn.init.xavier_normal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.00)
    return layer

class Tanh(nn.Module):
    """In-place tanh module."""

    def forward(self, input: th.Tensor) -> th.Tensor:
        """Forward pass for the in-place tanh activation function.

        Args:
            input (th.Tensor): Input tensor.

        Returns:
            Output tensor after applying in-place tanh.
        """
        return th.tanh_(input)

# Useful for picking up the right activation in the networks
th_act_fns = {
    'tanh': Tanh(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU()
}

