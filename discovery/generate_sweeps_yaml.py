import argparse
import pathlib
from copy import deepcopy
from distutils.util import strtobool

import wandb
from ruamel.yaml import YAML

SCRIPT_DIR = pathlib.Path(__file__).parent.parent / 'discovery'  # Scripts live here
SWEEPS_DIR = SCRIPT_DIR / 'sweep_yaml'
GEN_DIR = SCRIPT_DIR / 'gen_commands'
GEN_DIR.mkdir(parents=True, exist_ok=True)
BASIC_FILENAME = 'commands'

def wandb_sweep_to_textline(sweep):
    """Convert wandb sweep to flat list of commands with all arguments"""
    from itertools import product
    conf = deepcopy(sweep)
    script = conf['program']  # Same script for all
    params = conf['parameters']
    base_line = ['python', script]
    single_value_params = {k: v['value'] for k, v in params.items() if 'value' in v}
    single_value_params = {k: v['values'][0] for k, v in params.items() if 'values' in v and len(v['values']) == 1}
    base_line.extend([f'--{p} {v}' for p, v in single_value_params.items()])
    multiple_value_params = {k: v['values'] for k, v in params.items() if k not in single_value_params.keys()}
    keys, values = zip(*multiple_value_params.items())
    extensions = [dict(zip(keys, v)) for v in product(*values)]
    full_scripts = [' '.join(base_line + [f'--{p} {v}' for p,v in d.items()]) +'\n' for d in extensions]
    return full_scripts

def generate_combined_text_sweeps(c_confs, args):
    """Generate full text strings that can be run directly"""
    filename = BASIC_FILENAME + '_w_args.txt'
    mode = 'wt' if not args.append else 'at'
    f = (GEN_DIR / filename).open(mode)
    full_sweep_config = deepcopy(c_confs)
    lines = wandb_sweep_to_textline(full_sweep_config)  # Convert to list of full-text arguments
    f.writelines(lines)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default='dqn_sweep', 
                        help="Name of sweep file.")
    parser.add_argument('--append', type=lambda x:bool(strtobool(x)), default=True, 
                        help="Append to existing file")
    
    args, _ = parser.parse_known_args()
    base_parser = YAML(typ='safe')
    sweep_files = SWEEPS_DIR.glob(args.config_file + '.yaml')

    for path in sweep_files:
        conf = base_parser.load(path)
        generate_combined_text_sweeps(conf, args)
