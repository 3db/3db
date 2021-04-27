"""
threedb.controls.blender.orientation
====================================

Cover the subject object with another object. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/orientation.yaml>`_. 
"""

from typing import Any, Dict
import numpy as np
from threedb.controls.base_control import PreProcessControl


class OrientationControl(PreProcessControl):
    """This control changes the orientation of the object (i.e. the rotation).

    Continuous Parameters:

    - ``rotation_x``: The x component of the Eulerian rotation (range: ``[-pi, pi]``)
    - ``rotation_y``: The y component of the Eulerian rotation (range: ``[-pi, pi]``)
    - ``rotation_z``: The z component of the Eulerian rotation (range: ``[-pi, pi]``)

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/orientation/images/image_1.png
            :width: 100
            :group: orientation

        .. thumbnail:: /_static/logs/orientation/images/image_2.png
            :width: 100
            :group: orientation

        .. thumbnail:: /_static/logs/orientation/images/image_3.png
            :width: 100
            :group: orientation

        .. thumbnail:: /_static/logs/orientation/images/image_4.png
            :width: 100
            :group: orientation

        .. thumbnail:: /_static/logs/orientation/images/image_5.png
            :width: 100
            :group: orientation

        Varying the orientation across all parameters.
    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'rotation_x': (-np.pi, np.pi),
            'rotation_y': (-np.pi, np.pi),
            'rotation_z': (-np.pi, np.pi),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Rotates the object according to the given parameters

        Parameters
        ----------
        context : Dict[str, Any]
            The scene context object
        control_args : Dict[str, Any]
            The parameters for this control, ``rotation_x``, ``rotation_y``, and
            ``rotation_z``. See the class docstring for their documentation.
        """
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        obj = context['object']
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (control_args['rotation_x'],
                              control_args['rotation_y'],
                              control_args['rotation_z'])

    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = OrientationControl
