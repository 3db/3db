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
            'rotation_X': (-np.pi, np.pi),
            'rotation_Y': (-np.pi, np.pi),
            'rotation_Z': (-np.pi, np.pi),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Rotates the object according to the given parameters

        Parameters
        ----------
        context : Dict[str, Any]
            The scene context object
        control_args : Dict[str, Any]
            The parameters for this control, ``rotation_X``, ``rotation_Y``, and
            ``rotation_Z``. See the class docstring for their documentation.
        """
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        obj = context['object']
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (control_args['rotation_X'],
                              control_args['rotation_Y'],
                              control_args['rotation_Z'])

    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = OrientationControl
