import numpy as np
from sandbox.controls.base_control import BaseControl

class PoseControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'rotation_X': (-np.pi, np.pi),
        'rotation_Y': (-np.pi, np.pi),
        'rotation_Z': (-np.pi, np.pi),
    }

    discrete_dims = {}

    def apply(self, context, rotation_X, rotation_Y, rotation_Z):
        import mathutils
        import bpy

        # eul = mathutils.Euler((rotation_X, rotation_Y, rotation_Z), 'XYZ')
        ob = context['object']
        ob.rotation_euler = (rotation_X, rotation_Y, rotation_Z)
        # ob.rotation_quaternion = eul.to_quaternion()

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

BlenderControl = PoseControl
