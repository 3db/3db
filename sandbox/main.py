import argparse
import yaml
from glob import glob
from os import path

import sandbox
from sandbox.scheduling.dynamic_scheduler import schedule_work
from sandbox.scheduling.policy_controller import PolicyController
from sandbox.scheduling.SearchSpace import SearchSpace
from sandbox.utils import init_control
from sandbox.log import JSONLogger, TbLogger, ImageLogger


parser = argparse.ArgumentParser(
    description='Run a synthetic-sandbox experiment')
parser.add_argument('environment_folder', type=str,
                    help='folder containing all the environment (.blend)')

parser.add_argument('model_folder', type=str,
                    help='folder containing all models (.blend files)')

parser.add_argument('config_file', type=str,
                    help='Config file describing the experiment')

parser.add_argument('--logdir', type=str, default=None,
                    help='Log information about each sample into a file')

parser.add_argument('port', type=int,
                    help='The port used to listen for rendering workers')


DEFAULT_RENDER_ARGS = {
    'resolution': 256,
    'samples': 256,
}


if __name__ == '__main__':
    args = parser.parse_args()

    all_envs = [path.basename(x) for x in glob(path.join(args.environment_folder, '*.blend'))]
    all_models = [path.basename(x) for x in glob(path.join(args.model_folder, '*.blend'))]

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
        controls = [init_control(x) for x in config['controls']]
        search_space = SearchSpace(controls)
        continuous_dim, discrete_sizes = search_space.generate_description()

        policy_controllers = []

        json_logger = JSONLogger(path.join(args.logdir, 'details.log'))
        tb_logger = TbLogger(args.logdir)
        image_logger = ImageLogger(path.join(args.logdir, 'images'))
        
        json_logger.start()
        tb_logger.start()
        image_logger.start()

        for env in all_envs:
            env = env.split('/')[-1]
            for model in all_models:
                model = model.split('/')[-1]
                policy_controllers.append(
                    PolicyController(env, search_space, model, {
                        'continuous_dim': continuous_dim,
                        'discrete_sizes': discrete_sizes,
                        **config['policy']}, json_logger, tb_logger, image_logger))

        schedule_work(policy_controllers, args.port, all_envs, all_models,
                      render_args, config['inference'])

