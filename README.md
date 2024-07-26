# RL2Grid

RL2Grid is a benchmark representing realistic power grid operations aimed at fostering the maturity of RL methods. This work builds upon Grid2Op, a power grid simulation framework developed by RTE France, and enhances its usability as a benchmark for RL practitioners by providing standardized tasks, state and action spaces, and rewards within a common Gymnasium-based interface. RL2Grid extends the well-known [CleanRL codebase](https://github.com/vwxyzjn/cleanrl) to include flexible configurations for algorithm implementation details.

The user must set up [wandb](https://wandb.ai/home) to run and log the results of the experiments.

## Installation

1. Download [Miniconda](https://docs.anaconda.com/free/miniconda/) for your system.
2. Install Miniconda
3. Go to the RL2Grid main folder
    ```bash
    cd RL2Grid
    ```
3. Set-Up conda environment:
    ```bash
    conda env create -f conda_env.yml
    ```
4. Activate the conda environment
    ```bash
    conda activate rl2grid
    ```
4. Set-Up RL2Grid:
    ```bash
    pip install .
    ```

## Usage

Run the *main.py* with the desired set of parameters and task configuration. There are several parameters useful to log the results of a training run (see *main.py* argument parser as well as the algorithm-specific parameters under *alg/<algorithm>/config.py*

## First execution 

In order the run an experiment using asynchronous (vectorized) environments, the underlying Grid2Op framework requires generating and save an instance of the environment. To do so, you should run the *main.py* and set *generate_class* to True. After that, you can run an experiment in the specified environment by running the *main.py* with the desired hyperparameters.