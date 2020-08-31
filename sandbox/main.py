import argparse

import yaml
from glob import glob
from os import path
import importlib


parser = argparse.ArgumentParser(
    description='Run a synthetic-sandbox experiment')

parser.add_argument('environment_folder', type=str,
                    help='folder containing all the environment (.blend)')

parser.add_argument('model_folder', type=str,
                    help='folder containing all models (.blend files)')

parser.add_argument('config_file', type=str,
                    help='Config file describing the experiment')


def init_module(description):
    args = {k: v for (k, v) in description.items() if k != 'module'}
    module = importlib.import_module(description['module'])
    try:
        return module.Control(**args)
    except AttributeError:
        return module.Policy(**args)



class SearchSpace:

    def __init__(self, controls):
        self.controls = controls

        continuous_args = []
        discrete_args = []

        for control in self.controls:
            for continous_arg in control.continuous_dims:
                continuous_args.append((control, continous_arg))
            for k, v in control.discrete_dims.items():
                discrete_args.append((control, k, v))

        self.continuous_args = continuous_args
        self.discrete_args = discrete_args

    def generate_description(self):
        return len(self.continuous_args), [x[2] for x in self.discrete_args]

    def generate_log(self, packed_continuous, packed_discrete):
        pass

    def unpack(self, packed_continuous, packed_discrete):
        pass


class PolicyController:

    def __init__(self, env_file, model_name, policy_args):
        self.env_file = env_file,
        self.model_name = model_name
        self.policy = init_module(policy_args)


if __name__ == '__main__':
    args = parser.parse_args()

    all_envs = glob(path.join(args.environment_folder, '*.blend'))
    all_models = glob(path.join(args.model_folder, '*.blend'))

    with open(args.config_file) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
        print(config)
        assert 'policy' in config, 'Missing policy in config file'
        assert 'controls' in config, 'Missing control list in config file'
        # policy_module = importlib.import_module(config['policy']['module'])
        controls = [init_module(x) for x in config['controls']]
        search_space = SearchSpace(controls)
        continuous_dim, discrete_sizes = search_space.generate_description()

        policy_controllers = []

        for env in all_envs:
            for model in all_models:
                policy_controllers.append(PolicyController(env, model, {
                    'continuous_dim': continuous_dim,
                    'discrete_sizes': discrete_sizes,
                    **config['policy']}))

