"""
threedb.scheduling.search_space
===============================

This file includes the `SearchSpace` class which given a list of control and their argements, 
creates the corresponding search space of the cross product of all controls.
"""

from ast import literal_eval
import numpy as np


class SearchSpace:

    def __init__(self, controls):
        self.controls = controls

        continuous_args = []
        discrete_args = {}
        set_args = []

        for control in self.controls:
            tpe = type(control)
            name = f"{tpe.__name__}"
            for continous_arg, value_range in control.continuous_dims.items():
                if isinstance(value_range, str):
                    value_range = literal_eval(value_range)
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
                print(v)
                if not isinstance(v, (tuple, list)):
                    v = [v]
                if len(v) == 1:
                    set_args.append((name, k, v[0]))
                else:
                    discrete_args[(name, k)] = v

        self.continuous_args = continuous_args
        self.discrete_args = discrete_args
        self.set_args = set_args
        print(discrete_args)

    def generate_description(self):
        return len(self.continuous_args), [len(x) for x in self.discrete_args.values()]

    def generate_log(self, packed_continuous, packed_discrete):
        pass

    def unpack(self, packed_continuous, packed_discrete):
        result = {}
        for (control_name, attr_name, (start, end)), value in zip(self.continuous_args, packed_continuous):
            result[(control_name, attr_name)] = value * (end - start) + start

        for ((control_name, attr_name), values) , ix in zip(self.discrete_args.items(), packed_discrete):
            result[(control_name, attr_name)] = values[ix]

        for (control_name, attr_name, value) in self.set_args:
            result[(control_name, attr_name)] = value

        order_controls = [(type(x).__module__, type(x).__name__) for x in self.controls]

        return result, order_controls
