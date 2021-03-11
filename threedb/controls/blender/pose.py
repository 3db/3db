"""Defines the OrientationControl blender Control"""

import numpy as np
from threedb.controls.base_control import BaseControl


class OrientationControl(BaseControl):
    """This control changes the orientation of the object

    Note
    ----
    This control relies on Eulerian Rotations (X, Y, Z)

    Continuous Parameters
    ---------------------
    rotation_X
        The X component of the Eulerian rotation
    rotation_Y
        The Y component of the Eulerian rotation
    rotation_Z
        The Z component of the Eulerian rotation
    """

    kind = 'pre'

    continuous_dims = {
        'rotation_X': (-np.pi, np.pi),
        'rotation_Y': (-np.pi, np.pi),
        'rotation_Z': (-np.pi, np.pi),
    }

    discrete_dims = {}

    def apply(self, context, rotation_X, rotation_Y, rotation_Z):
        """Rotates the object according to the given parameters

        Parameters
        ----------
        context
            The scene context object
        rotation_X
            The X component of the Eulerian rotation
        rotation_Y
            The Y component of the Eulerian rotation
        rotation_Z
            The Z component of the Eulerian rotation
        """
        import bpy
        import mathutils

        ob = context['object']
        ob.rotation_euler = (rotation_X, rotation_Y, rotation_Z)


BlenderControl = OrientationControl
