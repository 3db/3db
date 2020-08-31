
class ExpositionControl:
    kind = 'post'
    continuous_dims = [
        'range_width',
        'range_offset'
    ]

    discrete_dims = {}

    def apply(self, continuous_args, discrete_args):
        pass



Control = ExpositionControl
