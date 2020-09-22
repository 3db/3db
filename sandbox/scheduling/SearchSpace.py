from ast import literal_eval

class SearchSpace:

    def __init__(self, controls):
        self.controls = controls

        continuous_args = []
        discrete_args = []
        set_args = []

        for control in self.controls:
            name = type(control).__name__
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
                discrete_args.append((name, k, v))

        self.continuous_args = continuous_args
        self.discrete_args = discrete_args
        self.set_args = set_args
        print(continuous_args)

    def generate_description(self):
        return len(self.continuous_args), [x[2] for x in self.discrete_args]

    def generate_log(self, packed_continuous, packed_discrete):
        pass

    def unpack(self, packed_continuous, packed_discrete):
        result = {}
        for (control_name, attr_name, _), value in zip(self.continuous_args, packed_continuous):
            result[f'{control_name}.{attr_name}'] = value

        for (control_name, attr_name, _), value in zip(self.discrete_args, packed_discrete):
            result[f'{control_name}.{attr_name}'] = value

        for (control_name, attr_name, value) in self.set_args:
            result[f'{control_name}.{attr_name}'] = value

        return result