"""
threedb.controls.blender.position
=================================

Control the object location. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/position.yaml>`_.
"""

from typing import Any, Dict
from threedb.controls.base_control import PreProcessControl
from .utils import post_translate, cleanup_translate_containers
from mathutils import Vector

class PositionControl(PreProcessControl):
    """This control changes the position of the object (i.e. translates it)

    .. note::
        The change in position is relative to the object's original position.

    Continuous Parameters:

    - ``offset_x``: Translation compnonent along the world's x-axis (range: ``[-1, 1]``)
    - ``offset_y``: Translation compnonent along the world's y-axis (range: ``[-1, 1]``)
    - ``offset_z``: Translation compnonent along the world's z-axis (range: ``[-1, 1]``)

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/position/images/image_1.png
            :width: 100
            :group: position

        .. thumbnail:: /_static/logs/position/images/image_2.png
            :width: 100
            :group: position

        .. thumbnail:: /_static/logs/position/images/image_3.png
            :width: 100
            :group: position

        .. thumbnail:: /_static/logs/position/images/image_4.png
            :width: 100
            :group: position

        .. thumbnail:: /_static/logs/position/images/image_5.png
            :width: 100
            :group: position

        Varying the position across the relevant parameters.
    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'offset_x': (-1., 1.),
            'offset_y': (-1., 1.),
            'offset_z': (-1., 1.),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        obj = context['object']
        post_translate(obj, Vector([control_args['offset_x'],
                                   control_args['offset_y'],
                                   control_args['offset_z']]))

    def unapply(self, context: Dict[str, Any]) -> None:
        cleanup_translate_containers(context['object'])

Control = PositionControl
