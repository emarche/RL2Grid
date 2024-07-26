#!/bin/bash
# Reserve 24 cores, 32G of total RAM, on the short partition for its max limit 24 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # 3
#SBATCH --cpus-per-task=24    # 8
#SBATCH --time=24:00:00
#SBATCH --mem=64G    # 128GB
#SBATCH --partition=short
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mail-user=e.marchesini@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Run our python script (1 task with 24 cores)
srun ~/.conda/envs/rl2grid_v1/bin/python run_sweeps_from_cmd_file.py
