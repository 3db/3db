"""
threedb.controls.blender.position
=================================
"""

from typing import Any, Dict
from threedb.controls.base_control import PreProcessControl
from .utils import post_translate, cleanup_translate_containers
from mathutils import Vector

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
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        obj = context['object']
        post_translate(obj, Vector([control_args['offset_X'],
                                   control_args['offset_Y'],
                                   control_args['offset_Z']]))

    def unapply(self, context: Dict[str, Any]) -> None:
        cleanup_translate_containers(context['object'])

Control = PositionControl
