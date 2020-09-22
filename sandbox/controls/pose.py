import numpy as np

class PoseControl:
    kind = 'pre'

    continuous_dims = {
        'rotation_X': (-np.pi, np.pi),
        'rotation_Y': (-np.pi, np.pi),
        'rotation_Z': (-np.pi, np.pi),
    }

    discrete_dims = {}

    def apply(self, continuous_args, discrete_args):
        pass


Control = PoseControl
