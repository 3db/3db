"""
threedb.controls.blender.position
=================================

[TODO]
"""

import numpy as np
from threedb.controls.base_control import BaseControl
from .utils import post_translate, cleanup_translate_containers

class PositionControl(BaseControl):
    """This control changes the position of the object (i.e. rotates it)

    Note
    ----
    The change in position is relative to the object's original position

    Continuous Parameters
    ---------------------
    offset_X
        Translation compnonent along the world's X-axis
    offset_Y
        Translation compnonent along the world's Y-axis
    offset_Z
        Translation compnonent along the world's Z-axis
    """

    kind = 'pre'

    continuous_dims = {
        'offset_X': (-1, 1),
        'offset_Y': (-1, 1),
        'offset_Z': (-1, 1),
    }

    discrete_dims = {}

    def apply(self, context, offset_X, offset_Y, offset_Z):
        """Rotates the object according to the given parameters
        Parameters
        ----------
        context
            The scene context object
        offset_X
            Translation compnonent along the world's X-axis
        offset_Y
            Translation compnonent along the world's Y-axis
        offset_Z
            Translation compnonent along the world's Z-axis
        """
        from mathutils import Vector

        ob = context['object']
        self.ob = ob
        post_translate(ob, Vector([offset_X, offset_Y, offset_Z]))

    def unapply(self, context):
        cleanup_translate_containers(self.ob)


BlenderControl = PositionControl
