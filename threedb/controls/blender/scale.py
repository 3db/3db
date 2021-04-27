"""
threedb.controls.blender.scale
==============================

Control the object scale. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/scale.yaml>`_.
"""

from typing import Any, Dict
from threedb.controls.base_control import PreProcessControl

class ObjScaleControl(PreProcessControl):
    """This control scales the object.

    Continuous Parameters:

    - ``factor``: scaling factor which takes any positive number. Setting the
        factor to 1 maintains the same object size. (default range: ``[0.25, 1]``)

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/scale/images/image_1.png
            :width: 100
            :group: scale

        .. thumbnail:: /_static/logs/scale/images/image_2.png
            :width: 100
            :group: scale

        .. thumbnail:: /_static/logs/scale/images/image_3.png
            :width: 100
            :group: scale

        .. thumbnail:: /_static/logs/scale/images/image_4.png
            :width: 100
            :group: scale

        .. thumbnail:: /_static/logs/scale/images/image_5.png
            :width: 100
            :group: scale
        
        Examples of scale at different levels.
    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'factor': (0.25, 1.),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """scales the object in the scene context by a factor `factor`

        Parameters
        ----------
        context
            The scene context object
        factor
            Scaling factor which takes any positive number. Setting the
            factor to 1 maintains the same object size.
        """
        self.ob = context['object']
        self.ob.scale = (control_args['factor'],) * 3

    def unapply(self, context):
        """Rescales the object to its original dimensions

        Parameters
        ----------
        context
            The scene context object
        """
        self.ob.scale = (1.,) * 3

Control = ObjScaleControl
