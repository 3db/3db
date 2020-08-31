
class PoseControl:
    kind = 'pre'
    continuous_dims = [
        'rotation_X',
        'rotation_Y',
        'rotation_Z',
    ]

    discrete_dims = {}

    def apply(self, continuous_args, discrete_args):
        pass


Control = PoseControl
