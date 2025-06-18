from common.imports import *

def get_alg_args() -> Namespace:
    """Parse command-line arguments for DQN.

    This function sets up and parses arguments for configuring the training and evaluation of a DQN agent.

    Returns:
        A namespace containing the parsed arguments.
    """
    parser = ap.ArgumentParser()

    parser.add_argument("--total-timesteps", type=int, default=5000000, help="Total timesteps for the experiment")
    parser.add_argument("--learning-starts", type=int, default=1000, help="When to start learning")
    parser.add_argument("--eval-freq", type=int, default=500, help="Total timesteps between deterministic evals")

    parser.add_argument('--layers', nargs='+', type=int, default=[256, 256], help='Actor network size')
    parser.add_argument('--act-fn', type=str, default='relu', help='Activation function')
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the Q-network")
    parser.add_argument('--train-freq', type=int, default=100, help='Training frequency in timesteps')

    parser.add_argument("--gamma", type=float, default=.9, help="Discount factor")
    parser.add_argument("--tau", type=float, default=1.0, help="Target network update rate")
    parser.add_argument('--tg-qnet-freq', type=int, default=500, help='Timesteps required to update the target network')
    parser.add_argument("--eps-start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon for exploration")
    parser.add_argument("--eps-decay-frac", type=float, default=0.4, help="Fraction of total-timesteps required to reach eps-end")

    parser.add_argument("--buffer-size", type=int, default=1000000, help="Replay memory buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size of sample from the replay memory")

    return parser.parse_known_args()[0]
