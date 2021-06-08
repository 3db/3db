"""
threedb.main
=============

The master of 3DB which is responsible for initializing controls, 
defining the search space, and scheduling the rendering tasks.
"""
import os
from typing import Type

# Making sure that we don't spawn some crazy multithreaded stuff
# All good since we do threading ourselves
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import importlib
import json
from pathlib import Path
from collections import defaultdict
from itertools import product
from os import makedirs, path

import torch as ch
import yaml
from tqdm import tqdm

from threedb.result_logging.logger_manager import LoggerManager
from threedb.scheduling.base_scheduler import Scheduler
from threedb.rendering.base_renderer import BaseRenderer
from threedb.scheduling.policy_controller import PolicyController
from threedb.scheduling.search_space import SearchSpace
from threedb.utils import CyclicBuffer, init_control
from typing import Dict, List, Any, Optional

parser = argparse.ArgumentParser(
    description='Run a 3DB experiment')

parser.add_argument('root_folder', type=str,
                    help='folder containing all data (models, environments, etc)')
parser.add_argument('config_file', type=str,
                    help='Config file describing the experiment')
parser.add_argument('output_dir', help='Where to store the output of the loggers',)
parser.add_argument('port', type=int,
                    help='The port used to listen for rendering workers')
parser.add_argument('--single-model', action='store_true',
                    help='If given, only do one model and one environment (for debugging)')
parser.add_argument('--max-concurrent-policies', '-m', type=int, default=10,
                    help='Maximum number of concurrent policies, can keep memory under control')

DEFAULT_RENDER_ARGS = {
    'engine': 'threedb.rendering.render_blender',
    'resolution': 256,
    'samples': 256,
    'with_uv': False,
    'with_depth': False,
    'with_segmentation': False,
    'max_depth': 10
}

def load_config(fpath: str) -> Dict[str, Any]:
    """Loads a configuration from a path to a YAML file, allows for inheritance
    between files using the ``base_config`` key.

    Parameters
    ----------
    path : str
        Path to the (YAML) config file
    """
    path = Path(fpath)
    with open(path) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
        if 'base_config' in config:
            base_config = load_config(path.parent / config['base_config'])
            base_config.update(config)
            return base_config
        else:
            return config

if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config(args.config_file)

    print(config)
    assert 'policy' in config, 'Missing policy in config file'
    assert 'controls' in config, 'Missing control list in config file'
    assert 'inference' in config, 'Missing `inference` key in config file'
    assert 'logging' in config, 'Missing `logging` key in config file'
    if 'render_args' in config:
        config['render_args'] = {**DEFAULT_RENDER_ARGS, **config['render_args']}

    rendering_module: Type[BaseRenderer] = getattr(importlib.import_module(config['render_args']['engine']), 'Renderer')

    all_models = rendering_module.enumerate_models(args.root_folder)
    all_envs = rendering_module.enumerate_environments(args.root_folder)

    if config['controls']:
        controls = [init_control(x, args.root_folder) for x in config['controls']]
    else:
        controls = []

    controls_args = defaultdict(dict)
    for i, control in enumerate(controls):
        tpe = type(control)
        name = f"{tpe.__name__}"
        control_config = config['controls'][i]
        if 'args' not in control_config:
            controls_args[name] = {} 
        else:
            controls_args[name] = control_config['args']
    config['controls'] = controls_args

    search_space = SearchSpace(controls)
    continuous_dim, discrete_sizes = search_space.generate_description()

    # Initialize the results buffer and register a single process
    result_buffer: CyclicBuffer = CyclicBuffer()
    policy_regid = result_buffer.register()  # Register a single policy for each output
    assert policy_regid == 1

    logger_manager = LoggerManager()
    logging_root = args.output_dir
    for module_path in config['logging']['logger_modules']:
        logger_module = importlib.import_module(module_path).Logger
        logger_manager.append(logger_module(logging_root, result_buffer, config))

    # Set up the policy controllers
    policy_controllers = set()
    for env, model in tqdm(list(product(all_envs, all_models)), desc="Init policies"):
        env = env.split('/')[-1]
        model = model.split('/')[-1]
        policy_args = {
            'continuous_dim': continuous_dim,
            'discrete_sizes': discrete_sizes,
            **config['policy']
        }
        controller = PolicyController(search_space, env, model, 
                            policy_args, logger_manager, result_buffer)
        policy_controllers.add(controller)
        if args.single_model: 
            break

    print("==> [Starting the scheduler]")
    s = Scheduler(args.port,
                    args.max_concurrent_policies, 
                    all_envs, 
                    all_models,
                    config,
                    policy_controllers,
                    result_buffer,
                    logger_manager)
    s.schedule_work()
