import argparse
import os
import pathlib
import subprocess
import sys
import time
from typing import List, Optional

from filelock import FileLock
from functools import wraps

SCRIPT_DIR = pathlib.Path(__file__).parent.parent / 'discovery'  # Scripts live here
GEN_DIR = SCRIPT_DIR / 'gen_commands'
SWEEP_FILE = GEN_DIR / 'commands_w_args.txt'
# os.environ["WANDB_START_METHOD"] = "thread"
if "SLURM_JOB_ID" not in os.environ: os.environ["SLURM_JOB_ID"] = 'none'
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def synchronized(lock_name, lock_path='.'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            lock_file = os.path.join(lock_path, f"{lock_name}.lock")
            with FileLock(lock_file):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@synchronized('sweep_access.lock', lock_path='.')
def load_sweeps_from_file(n: int = 32):
    """Load up to n sweeps from next sweep file"""
    if not SWEEP_FILE.exists():
        return []
    available_files = [SWEEP_FILE]
    cmds = []
    for file in available_files:
        if not file.exists():  # File was deleted between getting a listing and acquiring a lock
            print('No file')
            continue
        with file.open('r+') as f:
            lines = f.readlines()  # Load all the lines
            f.seek(0)
            f.truncate()
            n_lines = len(lines)  # How many lines are there total?
            lines_needed = n - len(cmds)  # Lines we currently need (could be < n)
            lines_to_read = min(n_lines, lines_needed)  # Lines to read from this file
            cmds.extend(lines[:lines_to_read])  # Append
            f.writelines(lines[lines_to_read:])
        if n_lines - lines_to_read <= 0:  # If we leave the file empty, delete it
            file.unlink(missing_ok=True)
            print('Deleted file')
        if len(cmds) >= n: break  # Have all the runs we need
        print(f'loaded {len(cmds)} param sets')
    # Otherwise, we run out of files, return
    return cmds



def launch_cmd_str(cmd_str: str):
    print(f'Job {os.environ["SLURM_JOB_ID"]} Running {cmd_str}')
    s = str.rstrip(cmd_str).split(' ')
    s[0] = sys.executable
    subprocess.run(s)  # Call and block, strip newline
    print(f'Job {os.environ["SLURM_JOB_ID"]} subprocess finished {cmd_str}')
    return 0


def wait_for_finish(procs: List[Optional[multiprocessing.Process]]):
    """Wait for all processes to finish, then quit"""
    print("Waiting for all processes to finish", flush=True)
    for i, p in enumerate(procs):
        print(f"Final process {i} wait")
        try:
            p.join()  # Wait forever
            p.close()
            procs[i] = None
        except:
            print(f"Process {i} is None")
            continue
    sys.exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cpus-per-task", type=int, default=8, help="Number of cpu per each training run.")
    args = parser.parse_args()

    # Initial load and dump
    if 'SLURM_CPUS_PER_TASK' not in os.environ: os.environ['SLURM_CPUS_PER_TASK'] = '4'  # If running locally, use 4 cores
    if 'SLURM_JOB_ID' not in os.environ: os.environ['SLURM_JOB_ID'] = 'None'  # If running locally, give an id
    n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])  # How many cores available
    n_cpus_per_task = args.n_cpus_per_task   # How many cores per script? 
    max_time = int((24 * 60 * 60) - (10 * 60 * 60)) if os.environ['SLURM_JOB_ID'] is not None else int(1e10)  # Time limit is 24 hours, don't want to get cut off
    
    t0 = time.time()
    n_processor_slots = int(n_cpus // n_cpus_per_task)
    scripts_to_run = load_sweeps_from_file(n_processor_slots)  # Load a specified run (or runs) from sweeps file (locks while we do it)

    if not scripts_to_run:  # Quit if none found
        print("No scripts at startup, exiting...")
        sys.exit(0)
    print(f'Found {n_processor_slots} scripts at start')

    idx = 0
    procs = []
    n_procs_running = 0
    """Start up processes"""
    while n_procs_running < n_processor_slots:
        scr = scripts_to_run[idx]
        p = multiprocessing.Process(target=launch_cmd_str, args=(scr, ))
        p.start()
        procs.append(p)
        n_procs_running += 1
        print(f'Job {os.environ["SLURM_JOB_ID"]} Process {n_procs_running} started')
        idx += 1

    # emarche: avoid running new jobs once the first ones are done
    wait_for_finish(procs)
    '''
    """Keep checking which ones are done"""
    finished = False
    while n_procs_running and not finished:
        for i, p in enumerate(procs):
            print(f"Checking process {i}")
            try:
                if not p.is_alive():  # Is this process finished (i.e., done with one particular run)
                    p.join(timeout=10)  # Give it 10 seconds to close gently
                    p.close()  # Make damn sure it's dead
                    n_procs_running -= 1  # If process finishes, subtract
                    procs[i] = None
                    # Only start new jobs if a) more are available b) we expect to be able to finish
                    if (time.time() - t0) < max_time:
                        scr = load_sweeps_from_file(1)
                        if not scr:
                            print(f'No more scripts available, waiting for terminations')
                            # wait_for_finish(procs)
                            finished = True
                            break
                        scr = scr[0]
                        print(f'Job {os.environ["SLURM_JOB_ID"]} Process {i} starting...')
                        p = multiprocessing.Process(target=launch_cmd_str, args=(scr,))  # Launch, put back in list
                        p.start()
                        procs[i] = p
                        n_procs_running += 1
                    else:
                        print("Early timeout, not starting new jobs")
                        finished = True
                        # wait_for_finish(procs)
                else:
                    continue  # process still working
            except:  # None object, has no join
                continue
        time.sleep(120)  # Wait a while before checking again

    if finished: wait_for_finish(procs)
    '''
    sys.exit(0)