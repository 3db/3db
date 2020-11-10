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
        bbox_corners = [ob.matrix_world @ Vector(corner) for corner in ob.bound_box]
        obj_min_z = min((bbc[2] for bbc in bbox_corners))
        # embed()
        ob.location.z += z_ground - obj_min_z

BlenderControl = PinToGroundControl
