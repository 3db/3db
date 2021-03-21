"""
threedb.controls.blender.position
=================================

[TODO]
"""

from typing import Any, Dict
import numpy as np
from threedb.controls.base_control import PreProcessControl
from .utils import post_translate, cleanup_translate_containers

class PositionControl(PreProcessControl):
    """This control changes the position of the object (i.e. rotates it)

    .. note::
        The change in position is relative to the object's original position.

    Continuous Parameters:

    - offset_X: Translation compnonent along the world's X-axis (range: [-1, 1])
    - offset_Y: Translation compnonent along the world's Y-axis (range: [-1, 1])
    - offset_Z: Translation compnonent along the world's Z-axis (range: [-1, 1])
    """
    def __init__(self, root_folder: str):
        continuous_dims = {
        'offset_X': (-1., 1.),
        'offset_Y': (-1., 1.),
        'offset_Z': (-1., 1.),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Rotates the object according to the given parameters.

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
        post_translate(ob, Vector([control_args['offset_X'],
                                   control_args['offset_Y'],
                                   control_args['offset_Z']]))

    def unapply(self, context: Dict[str, Any]) -> None:
        cleanup_translate_containers(self.ob)

Control = PositionControl
