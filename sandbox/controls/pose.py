import numpy as np

class PoseControl:
    kind = 'pre'

    continuous_dims = {
        'rotation_X': (-np.pi, np.pi),
        'rotation_Y': (-np.pi, np.pi),
        'rotation_Z': (-np.pi, np.pi),
    }

    discrete_dims = {}

    def apply(self, context, rotation_X, rotation_Y, rotation_Z):

        ob = context['object']
        ob.rotation_euler = (rotation_X, rotation_Y, rotation_Z)


Control = PoseControl
