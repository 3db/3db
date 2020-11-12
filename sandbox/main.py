import argparse
import yaml
from glob import glob
from os import path, makedirs
from collections import defaultdict
import importlib

import sandbox
from sandbox.scheduling.dynamic_scheduler import schedule_work
from sandbox.scheduling.policy_controller import PolicyController
from sandbox.scheduling.SearchSpace import SearchSpace
from sandbox.utils import init_control
from sandbox.log import JSONLogger, TbLogger, ImageLogger, LoggerManager


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
                    help='Which loggers to use. Comma dilimited list. e.g. JSONLogger,TbLogger,ImageLogger')


DEFAULT_RENDER_ARGS = {
    'engine': 'sandbox.rendering.blender',
    'resolution': 256,
    'samples': 256,
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
        rendering_engine = importlib.import_module(render_args['engine'])

        all_models = rendering_engine.enumerate_models(args.root_folder)
        all_envs = rendering_engine.enumerate_environments(args.root_folder)

        if config['controls']:
            controls = [init_control(x, args.root_folder, rendering_engine.NAME)
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

        logger_manager = LoggerManager()
        loggers_list = [logger for logger in args.loggers.split(',')]
        if "JSONLogger" in loggers_list:
            logger_manager.append(JSONLogger(path.join(args.logdir, 'details.log')))
        if "TbLogger" in loggers_list:
            logger_manager.append(TbLogger(args.logdir))
        if "ImageLogger" in loggers_list:
            imgdir = path.join(args.logdir, 'images')
            if not path.exists(imgdir):
                makedirs(imgdir)
            logger_manager.append(ImageLogger(imgdir))
        logger_manager.start()
        for env in all_envs:
            env = env.split('/')[-1]
            for model in all_models:
                model = model.split('/')[-1]
                # model = 'ba705749a39d4f5382b265c7b157e962.blend'
                policy_controllers.append(
                    PolicyController(env, search_space, model, {
                        'continuous_dim': continuous_dim,
                        'discrete_sizes': discrete_sizes,
                        **config['policy']}, logger_manager))
                # break
        schedule_work(policy_controllers, args.port, all_envs, all_models,
                      render_args, config['inference'], controls_args)

