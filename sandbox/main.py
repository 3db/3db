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
# from sandbox.scheduling.SearchSpace import SearchSpace
from sandbox.utils import BigChungusCyclicBuffer, init_control
from typing import Dict, List, Any, Optional

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

        # render_args = DEFAULT_RENDER_ARGS
        if 'render_args' in config:
            config['render_args'] = {**DEFAULT_RENDER_ARGS, **config['render_args']}
            # render_args.update(config['render_args'])

        # print("ARGS", render_args)
        rendering_module: Type[BaseRenderer] = getattr(importlib.import_module(config['render_args']['engine']), 'Renderer')

        all_models = rendering_module.enumerate_models(args.root_folder)
        all_envs = rendering_module.enumerate_environments(args.root_folder)

        if config['controls']:
            controls = [init_control(x, args.root_folder, rendering_module.NAME)
                        for x in config['controls']]
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

        with open(config['inference']['uid_to_targets'], 'r') as f:
            uid_to_targets = json.load(f)

        """
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
        """
        # big_chungus = BigChungusCyclicBuffer()
        # policy_regid = big_chungus.register()  # Register a single policy for each output
        # assert policy_regid == 1

        # Initialize the results buffer and register a single process
        result_buffer: BigChungusCyclicBuffer = BigChungusCyclicBuffer()
        policy_regid = result_buffer.register()  # Register a single policy for each output
        assert policy_regid == 1

        logger_manager = LoggerManager()
        loggers_list = [logger for logger in args.loggers.split(',')]
        print(loggers_list)
        if "JSONLogger" in loggers_list:
            logger_manager.append(JSONLogger(args.logdir, result_buffer, config))
        if "TbLogger" in loggers_list:
            logger_manager.append(TbLogger(args.logdir, result_buffer, config))
        if "ImageLogger" in loggers_list:
            print("STARTING IMAGE LOGGER")
            imgdir = os.path.join(args.logdir, 'images')
            if not os.path.exists(imgdir):
                os.makedirs(imgdir)
            logger_manager.append(ImageLogger(imgdir, result_buffer, config))
        logger_manager.start()

        print("==> [Starting the scheduler]")
        schedule_work(args.port,
                      args.max_concurrent_policies, 
                      all_envs, 
                      all_models,
                      controls,
                      config,
                      result_buffer,
                      logger_manager,
                      args.single_model)

        # Warning the logger that we are done
        logger_manager.log(None)
        print("==> [Waiting for any pending logging]")
        # We have to wait until it has processed everything left in the queue
        logger_manager.join()
        print("==> [Have a nice day!]")

