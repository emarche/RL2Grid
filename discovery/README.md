# Discovery scripts

This is the current setup for running experiments on a SLURM-based cluster. It involves a few steps to set up and run the desired experiments.

- Modify the "singlejob_submit.sh" script as needed. This is used to get a node and run the Python scripts that have been previously generated (see next two steps). You have to modify the email (so you'll be notified when a failure happens or when the job ends), and the path to your conda environment in the last line.

- YAML files in "sweep_yaml" are used to set up the parameters we want to run for the experiments (see the included ppo_paper.yaml as a representative example). You just have to set the desired parameters from the ones available in the algorithm/env config files. It is important to:
    1. Set the n-threads parameter to decide how many cpus each experiment will use from the ones reserved in the singlejob_submit.sh script (e.g., in the current setups, you're reserving a node with 24 cpus in singlejob_submit and setting n-threads to 8, so you'll run 24/8 scripts on the node).
    2. Set the same n-threads number in "run_sweeps_from_cmd_file.py" as a default parameter at line 88.

- Run "python generate_sweeps_yaml.py --config-file==<name of the yaml sweep file> --append=False": this is used to generate a .txt file in "gen_commands" with all the python scripts to run (e.g., for the previous ppo file, you'll run "python generate_sweeps_yaml.py --config-file==ppo_paper --append=False").

- Finally, execute "multijob_submit.sh" to run the experiments. Set the desired number of N nodes you want to run (this will run N singlejob_submit instances)

Additional notes:
    - Remember to set the appropriate "time-limit" parameter in the "main.py;" this is the time limit for the runs. If "checkpoint" is set to True, you'll find the saved checkpoints for each run in "discovery > checkpoint." Upon termination, we also try to sync wandb runs automatically, so you might want to handle this if you're not using wandb.