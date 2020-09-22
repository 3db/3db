import argparse
import yaml
from glob import glob
from os import path

import sandbox
from sandbox.scheduling.dynamic_scheduler import schedule_work
from sandbox.scheduling.policy_controller import PolicyController
from sandbox.utils import init_module


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


import collections
class SearchSpace:

    def __init__(self, controls):
        self.controls = controls

        continuous_args = []
        discrete_args = []
        set_args = []

        for control in self.controls:
            name = type(control).__name__
            for continous_arg, value_range in control.continuous_dims.items():
                try:
                    if len(value_range) != 2:
                        raise ValueError(
                            'range {value_range} for {control} should have length 2')
                    continuous_args.append((name, continous_arg, value_range))
                except TypeError:
                    # This is not a range but a constant value
                    # We consider this paramter to be set and not searched over
                    set_args.append((name, continous_arg, value_range))
            for k, v in control.discrete_dims.items():
                discrete_args.append((name, k, v))

        self.continuous_args = continuous_args
        self.discrete_args = discrete_args
        self.set_args = set_args

    def generate_description(self):
        return len(self.continuous_args), [x[2] for x in self.discrete_args]

    def generate_log(self, packed_continuous, packed_discrete):
        pass

    def unpack(self, packed_continuous, packed_discrete):
        result = {}
        for (control_name, attr_name), value in zip(self.continuous_args, packed_continuous):
            result[f'{control_name}.{attr_name}'] = value

        for (control_name, attr_name, _), value in zip(self.discrete_args, packed_discrete):
            result[f'{control_name}.{attr_name}'] = value

        return result



if __name__ == '__main__':
    args = parser.parse_args()

    all_envs = [path.basename(x) for x in glob(path.join(args.environment_folder, '*.blend'))]
    all_models = [path.basename(x) for x in glob(path.join(args.model_folder, '*.blend'))]

    with open(args.config_file) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
        print(config)
        assert 'policy' in config, 'Missing policy in config file'
        assert 'controls' in config, 'Missing control list in config file'
        controls = [init_module(x) for x in config['controls']]
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

        schedule_work(policy_controllers, args.port, all_envs, all_models)

