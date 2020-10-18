import numpy as np
import mathutils

class PoseControl:
    kind = 'pre'

    continuous_dims = {
        'rotation_X': (-np.pi, np.pi),
        'rotation_Y': (-np.pi, np.pi),
        'rotation_Z': (-np.pi, np.pi),
    }

    discrete_dims = {}

    def apply(self, context, rotation_X, rotation_Y, rotation_Z, **kwargs):

        eul = mathutils.Euler((rotation_X, rotation_Y, rotation_Z), 'XYZ')
        ob = context['object']
        ob.rotation_quaternion = eul.to_quaternion()

Control = PoseControl
