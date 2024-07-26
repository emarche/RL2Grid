from common.imports import *

def get_alg_args() -> Namespace:
    """Parse command-line arguments for TD3.

    This function sets up and parses arguments for configuring the training and evaluation of a TD3 agent.

    Returns:
        A namespace containing the parsed arguments.
    """
    parser = ap.ArgumentParser()

    parser.add_argument("--total-timesteps", type=int, default=int(1e6), help="Total timesteps for the experiment")
    parser.add_argument("--learning-starts", type=int, default=int(5e3), help="When to start learning")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Total timesteps between deterministic evals")

    parser.add_argument('--actor-layers', nargs='+', type=int, default=[32, 32], help='Actor network size')
    parser.add_argument('--critic-layers', nargs='+', type=int, default=[32, 32], help='Critic network size')
    parser.add_argument('--actor-act-fn', type=str, default='relu', help='Actor activation function')
    parser.add_argument('--critic-act-fn', type=str, default='relu', help='Critic activation function')
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Learning rate for the actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="Learning rate for the critic")
    parser.add_argument('--actor-train-freq', type=int, default=2, help='Training frequency in timesteps')
    parser.add_argument('--tg-freq', type=int, default=1000, help='Timesteps required to update the target networks')
    parser.add_argument('--train-freq', type=int, default=100, help='Training frequency in timesteps')

    parser.add_argument("--gamma", type=float, default=.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="Target smoothing coefficient")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="Scale for the policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="Scale for the exploration noise")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="Noise clip for target policy smoothing regularization")
    
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="Replay memory buffer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size of sample from the replay memory")
    
    return parser.parse_known_args()[0]
