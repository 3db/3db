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
from collections import defaultdict
from itertools import product
from os import makedirs, path

import torch as ch
import yaml
from tqdm import tqdm

from sandbox.log import ImageLogger, JSONLogger, LoggerManager, TbLogger
from sandbox.scheduling.dynamic_scheduler import schedule_work
from sandbox.rendering.base_renderer import BaseRenderer
from sandbox.scheduling.policy_controller import PolicyController
from sandbox.scheduling.SearchSpace import SearchSpace
from sandbox.utils import BigChungusCyclicBuffer, init_control

parser = argparse.ArgumentParser(
    description='Run a synthetic-sandbox experiment')

parser.add_argument('root_folder', type=str,
                    help='folder containing all data (models, environments, etc)')
parser.add_argument('config_file', type=str,
                    help='Config file describing the experiment')
parser.add_argument('--logdir', type=str, default=None,
                    help='Log information about each sample into a folder')
parser.add_argument('port', type=int,
                    help='The port used to listen for rendering workers')
parser.add_argument('--loggers', type=str, default='JSONLogger,TbLogger',
                    help='Loggers to use (comma-delimited), e.g. JSONLogger,TbLogger,ImageLogger')
parser.add_argument('--single-model', action='store_true',
                    help='If given, only do one model and one environment (for debugging)')
parser.add_argument('--max-concurrent-policies', '-m', type=int, default=10,
                    help='Maximum number of concurrent policies, can keep memory under control')

DEFAULT_RENDER_ARGS = {
    'engine': 'sandbox.rendering.blender',
    'resolution': 256,
    'samples': 256,
    'with_uv': False,
    'with_depth': False,
    'with_segmentation': False,
    'max_depth': 10
}

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config_file) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
        print(config)
        assert 'policy' in config, 'Missing policy in config file'
        assert 'controls' in config, 'Missing control list in config file'
        assert 'inference' in config, 'Missing `inference` key in config file'
        render_args = DEFAULT_RENDER_ARGS
        if 'render_args' in config:
            render_args.update(config['render_args'])

        print("ARGS", render_args)
        rendering_module: Type[BaseRenderer] = getattr(importlib.import_module(render_args['engine']), 'Renderer')

        all_models = rendering_module.enumerate_models(args.root_folder)
        all_envs = rendering_module.enumerate_environments(args.root_folder)

        if config['controls']:
            controls = [init_control(x, args.root_folder, rendering_module.NAME)
                        for x in config['controls']]
        else:
            controls = []
        controls_args = defaultdict(dict)
        for i,control in enumerate(controls):
            tpe = type(control)
            name = f"{tpe.__name__}"
            control_config = config['controls'][i]
            if 'args' not in control_config:
                controls_args[name] = {} 
            else:
                controls_args[name] = control_config['args']

        search_space = SearchSpace(controls)
        continuous_dim, discrete_sizes = search_space.generate_description()

        policy_controllers = []

        with open(config['inference']['uid_to_targets'], 'r') as f:
            uid_to_targets = json.load(f)

        # We need to know the resolution and number of classes to allocate
        # the memory beforehand and share it with other processes
        imsize = [render_args['resolution'], render_args['resolution']]
        buffer_channels = {'rgb': ([3, *imsize], ch.float32)}
        # render_channels = [('rgb', 3, ch.float32)]
        if render_args['with_uv']:
            buffer_channels['uv'] = ([3, *imsize], ch.float32)
            # render_channels.append(('uv', 3, ch.float32))
        if render_args['with_depth']:
            buffer_channels['depth'] = ([3, *imsize], ch.float32)
            # render_channels.append(('depth', 3, ch.float32))
        if render_args['with_segmentation']:
            buffer_channels['segmentation'] = ([1, *imsize], ch.int32)
            # render_channels.append(('segmentation', 1, ch.int32))
        buffer_channels['output'] = (config['inference']['output_shape'], ch.float32)
        buffer_channels['is_correct'] = ([], ch.bool)

        big_chungus = BigChungusCyclicBuffer(buffer_channels)
        policy_regid = big_chungus.register()  # Register a single policy for each output
        assert policy_regid == 1

        logger_manager = LoggerManager()
        loggers_list = [logger for logger in args.loggers.split(',')]
        if "JSONLogger" in loggers_list:
            logger_manager.append(JSONLogger(args.logdir, big_chungus, config))
        if "TbLogger" in loggers_list:
            logger_manager.append(TbLogger(args.logdir, big_chungus, config))
        if "ImageLogger" in loggers_list:
            imgdir = path.join(args.logdir, 'images')
            if not path.exists(imgdir):
                makedirs(imgdir)
            logger_manager.append(ImageLogger(imgdir, big_chungus, config))
        logger_manager.start()

        for env, model in tqdm(list(product(all_envs, all_models)), desc="Init policies"):
            env = env.split('/')[-1]
            model = model.split('/')[-1]
            policy_controllers.append(
                PolicyController(env, search_space, model, {
                    'continuous_dim': continuous_dim,
                    'discrete_sizes': discrete_sizes,
                    **config['policy']}, logger_manager, big_chungus))
            if args.single_model: 
                break

        print("==>[Starting the scheduler]")
        import multiprocessing
        schedule_work(policy_controllers, args.port,
                      args.max_concurrent_policies, all_envs, all_models,
                      render_args, config['inference'], controls_args,
                      config['evaluation'], big_chungus)

        # Warning the logger that we are done
        logger_manager.log(None)
        print("==> [Waiting for any pending logging]")
        # We have to wait until it has processed everything left in the queue
        logger_manager.join()
        print("==> [Have a nice day!]")

