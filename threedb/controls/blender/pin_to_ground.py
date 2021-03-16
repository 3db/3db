import numpy as np
from threedb.controls.base_control import BaseControl

try:
    from threedb.controls.blender.blender_utils import (
        cleanup_translate_containers, post_translate)
except:
    pass


class PinToGroundControl(BaseControl):
    """Control that moves an object vertically to touch the ground
    Useful when you want a slightly more realistic rendering 
    (i.e. avoid flying objects)

    Continuous Dimensions
    ---------------------
    z_ground
        the Z-coordinate of the surface underneath the object to which
        you want to pin the object. Takes any real number.

    Note
    ----
    This control shall be used after the `PositionControl` and 
    `OrientationControl` controls, i.e., move the object to a location of
    interest, then drag it to the ground under that location. 

    """
    kind = 'pre'

    continuous_dims = {
        'z_ground': (0, 1),
    }

    discrete_dims = {}

    def apply(self, context, z_ground):
        """Pins the object to the ground

        Parameters
        ----------
        context
            The scene context object
        z_ground
            the Z-coordinate of the surface underneath the object to which
            you want to pin the object. Takes any real number.
        """
        import bpy
        from bpy import context as C
        from mathutils import Vector
        ob = context["object"]
        bpy.context.view_layer.update()
        obj_min_z = min((ob.matrix_world @ x.co)[2] for x in ob.data.vertices)
        post_translate(ob, Vector([0, 0, z_ground - obj_min_z]))
        self.ob = ob

    def unapply(self, context):
        cleanup_translate_containers(self.ob)



BlenderControl = PinToGroundControl