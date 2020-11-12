import numpy as np
from sandbox.controls.base_control import BaseControl
from IPython import embed

try:
    from sandbox.controls.blender.blender_utils import (
        cleanup_translate_containers, post_translate)
except:
    pass


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
        bpy.context.view_layer.update()
        obj_min_z = min((ob.matrix_world @ x.co)[2] for x in ob.data.vertices)
        post_translate(ob, Vector([0, 0, z_ground - obj_min_z]))
        self.ob = ob

    def unapply(self):
        cleanup_translate_containers(self.ob)



BlenderControl = PinToGroundControl
