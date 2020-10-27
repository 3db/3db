
class ExpositionControl:
    kind = 'post'
    continuous_dims = {
        'range_width': (-1, 1),
        'range_offset': (-1, 1)
    }

    discrete_dims = {}

    def apply(self, continuous_args, discrete_args):
        pass



ControlBlender = ExpositionControl
