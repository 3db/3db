import argparse
import yaml
from glob import glob
from os import path

import sandbox
from sandbox.scheduling.dynamic_scheduler import schedule_work
from sandbox.scheduling.policy_controller import PolicyController
from sandbox.scheduling.SearchSpace import SearchSpace
from sandbox.utils import init_control


parser = argparse.ArgumentParser(
    description='Run a synthetic-sandbox experiment')
parser.add_argument('environment_folder', type=str,
                    help='folder containing all the environment (.blend)')

parser.add_argument('model_folder', type=str,
                    help='folder containing all models (.blend files)')

parser.add_argument('config_file', type=str,
                    help='Config file describing the experiment')

parser.add_argument('port', type=int,
                    help='The port used to listen for rendering workers')


DEFAULT_RENDER_ARGS = {
    'resolution': 256,
    'samples': 256,
    'image_format': 'png'
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
        render_args = DEFAULT_RENDER_ARGS
        if 'render_args' in config:
            render_args.update(config['render_args'])

        print("ARGS", render_args)
        controls = [init_control(x) for x in config['controls']]
        search_space = SearchSpace(controls)
        continuous_dim, discrete_sizes = search_space.generate_description()

        policy_controllers = []

        for env in all_envs:
            env = env.split('/')[-1]
            for model in all_models:
                model = model.split('/')[-1]
                policy_controllers.append(
                    PolicyController(env, search_space, model, {
                        'continuous_dim': continuous_dim,
                        'discrete_sizes': discrete_sizes,
                        **config['policy']}))

        schedule_work(policy_controllers, args.port, all_envs, all_models,
                      render_args)

