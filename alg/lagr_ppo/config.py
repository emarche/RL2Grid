from common.imports import *
from common.utils import str2bool

def get_alg_args() -> Namespace:
    """Parse command-line arguments for Lagrangian PPO.

    This function sets up and parses arguments for configuring the training and evaluation of a Lagrangian PPO agent.

    Returns:
        A namespace containing the parsed arguments.
    """
    parser = ap.ArgumentParser()

    parser.add_argument("--total-timesteps", type=int, default=25000000, help="Total timesteps for the experiment")
    parser.add_argument("--n-steps", type=int, default=2000, help="Steps per policy rollout")    # 20k for 1 env
    parser.add_argument("--eval-freq", type=int, default=500, help="Total timesteps between deterministic evals")

    parser.add_argument('--actor-layers', nargs='+', type=int, default=[256, 256], help='Actor network size')
    parser.add_argument('--critic-layers', nargs='+', type=int, default=[256, 256], help='Critic network size')
    parser.add_argument('--actor-act-fn', type=str, default='tanh', help='Actor activation function')
    parser.add_argument('--critic-act-fn', type=str, default='tanh', help='Critic activation function')
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Learning rate for the actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Learning rate for the critic")
    parser.add_argument("--anneal-lr", type=str2bool, default=True, help="Toggles learning rate annealing")

    parser.add_argument("--gamma", type=float, default=.9, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=.95, help="Lambda for the genralized advantage estimation")

    parser.add_argument("--update-epochs", type=int, default=80, help="Number of update epochs")
    parser.add_argument("--n-minibatches", type=int, default=4, help="Number of minibatches")
    parser.add_argument("--max-grad-norm", type=float, default=10, help="Maximum norm for gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="Target KL divergence threshold")

    parser.add_argument("--norm-adv", type=str2bool, default=True, help="Toggles advantage normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="Surrogate clip coefficient")
    parser.add_argument("--clip-vfloss", type=str2bool, default=True, help="Toggles clip for value function loss")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")

    parser.add_argument("--cost-threshold", type=float, default=15.0, help="Cost threshold")
    parser.add_argument("--lag-mul", type=float, default=1., help="Initial value for the multiplier")
    parser.add_argument("--lag-lr", type=float, default=0.05, help="Learning rate for the multiplier")

    return parser.parse_known_args()[0]
