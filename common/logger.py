import os
import subprocess
from collections import deque 

import shutil

from .imports import *

class Logger:
    """Logger class for managing and logging metrics to WandB.

    Attributes:
        log_freq (int): Frequency of logging metrics.
        episodic_survival (Deque[float]): Deque to store episodic survival metrics.
        episodic_return (Deque[float]): Deque to store episodic return metrics.
        episodic_length (Deque[float]): Deque to store episodic length metrics.
        wb_mode (str): Mode for WandB (online/offline).
        wb_path (str): Path to WandB logging directory.
    """

    def __init__(self, run_name: str, args: Dict[str, Any], log_freq: int = 1) -> None:
        """Initialize the Logger with given parameters.

        Args:
            run_name: Name of the run for logging.
            args: Arguments containing configuration for logging.
            log_freq: Frequency of logging metrics.
        """
        self.log_freq = log_freq

        self.episodic_survival = deque(maxlen=log_freq)
        self.episodic_return = deque(maxlen=log_freq)
        self.episodic_length = deque(maxlen=log_freq)

        self.wb_mode = args.wandb_mode

        wb_path = wb.init(
            name=run_name,
            id=run_name,
            config=vars(args),
            mode=self.wb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            settings=wb.Settings(_disable_stats=True),
            resume=True
            #sync_tensorboard=True,
        )
        self.wb_path = os.path.split(wb_path.dir)[0]

    def store_metrics(self, global_step: int, avg_survival: float, avg_return: float) -> None:
        """Store the given metrics and log them if the log frequency is met.

        Args:
            global_step: Current global step of training.
            avg_survival: Average survival metric to be stored.
            avg_return: Average return metric to be stored.
        """
        self.episodic_survival.append(avg_survival)
        self.episodic_return.append(avg_return)
        if global_step % self.log_freq == 0: self.log_metrics(global_step)

    def log_metrics(self, global_step: int) -> None:
        """Log the stored metrics to WandB.

        Args:
            global_step: Current global step of training.
        """
        record = {
            "charts/episodic_survival": np.mean(self.episodic_survival),
            "charts/episodic_return": np.mean(self.episodic_return),
            "charts/global_step": global_step
        }
        wb.log(record, step=global_step)
    
    def close(self) -> None:
        """Close the logger and clean up resources."""
        if self.wb_path is not None and self.wb_mode == 'offline':
            wb.finish()
            subprocess.run(['wandb', 'sync', '--append', self.wb_path]) 
            shutil.rmtree(self.wb_path)   # Remove wandb run folder

