from collections import deque

from common.imports import *
from common.logger import Logger
from .utils import make_env

class Evaluator:
    """Evaluator class for evaluating a reinforcement learning model deterministically.

    Attributes:
        env (gym.Env): Vectorized environment for evaluation.
        max_steps (int): Maximum number of steps in an episode.
        logger (Logger): Logger for storing evaluation metrics.
        device (th.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    def __init__(self, args: Dict[str, Any], logger: Logger, device: th.device) -> None:
        """Initialize the Evaluator with the given arguments, logger, and device.

        Args:
            args: Arguments containing environment configuration.
            logger: Logger for storing evaluation metrics.
            device: Device to run the model on.
        """
        self.env = gym.vector.SyncVectorEnv([make_env(args, 0)])  # Initialize synchronized vector environment
        self.max_steps = self.env.envs[0].init_env.chronics_handler.max_episode_duration()  # Get max episode duration
        self.logger = logger  # Logger for evaluation metrics
        self.device = device  # Device for model inference

    def evaluate(self, glob_step: int, model: object, eval_ep: int = 10) -> None:
        """Evaluate the model over a specified number of episodes.

        Args:
            glob_step: Global step for logging purposes.
            model: Model to be evaluated.
            eval_ep: Number of episodes for evaluation.
        """
        obs, _ = self.env.reset()

        ep_survivals: Deque[float] = deque(maxlen=eval_ep)  # Queue to store survival rates of episodes
        ep_returns: Deque[float] = deque(maxlen=eval_ep)  # Queue to store returns of episodes
        
        while len(ep_survivals) < eval_ep:
            action = model.get_eval_action(th.tensor(obs).to(self.device)).detach().numpy()
            next_obs, _, _, _, info = self.env.step(action)

            # Record rewards for plotting purposes
            if "final_info" in info:
                for info in info['final_info']:
                    if info and "episode" in info:
                        ep_survivals.append(info['episode']['l'][0]/self.max_steps)
                        ep_returns.append(info['episode']['r'][0])
            
            obs = next_obs

        # Calculate average survival rate and return over the evaluated episodes
        avg_survival = sum(ep_survivals)/eval_ep
        avg_return = sum(ep_returns)/eval_ep

        # Log the metrics if logger is available
        if self.logger: self.logger.store_metrics(glob_step, avg_survival, avg_return)

        print(f"Eval at step {glob_step}, survival={avg_survival*100:.3f}%, return={avg_return}")

