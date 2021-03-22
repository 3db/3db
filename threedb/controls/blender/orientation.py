"""
threedb.controls.blender.pose
==============================

[TODO]
"""

from typing import Any, Dict
import numpy as np
from threedb.controls.base_control import PreProcessControl


class OrientationControl(PreProcessControl):
    """This control changes the orientation of the object

    .. note::

        This control relies on Eulerian Rotations (X, Y, Z)

    Continuous Parameters:
    
    - rotation_X: The X component of the Eulerian rotation (range: [-pi, pi])
    - rotation_Y: The Y component of the Eulerian rotation (range: [-pi, pi])
    - rotation_Z: The Z component of the Eulerian rotation (range: [-pi, pi])
    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'rotation_X': (-np.pi, np.pi),
            'rotation_Y': (-np.pi, np.pi),
            'rotation_Z': (-np.pi, np.pi),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context, rotation_X, rotation_Y, rotation_Z):
        pass 

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
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
        ob = context['object']
        ob.rotation_euler = (control_args['rotation_X'],
                             control_args['rotation_Y'],
                             control_args['rotation_Z'])
    
    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = OrientationControl
