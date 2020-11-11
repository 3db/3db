import numpy as np
from sandbox.controls.base_control import BaseControl
from IPython import embed

class PinToGroundControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'z_ground': (0, 1),
    }

    discrete_dims = {}

    def apply(self, context, z_ground):
        import bpy
        from bpy import context as C
        from mathutils import Vector
        ob = context["object"]
        obj_min_z = min((ob.matrix_world @ x.co)[2] for x in ob.data.vertices)
        ob.location.z += z_ground - obj_min_z

BlenderControl = PinToGroundControl
