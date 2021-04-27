"""
threedb.controls.blender.pin_to_ground
======================================

Control where the bottom bounding box of an object lies on the z axis. An example
config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/pin_to_ground.yaml>`_.
"""
from typing import Any, Dict
from ...try_bpy import bpy
from mathutils import Vector
from threedb.controls.base_control import PreProcessControl
from .utils import cleanup_translate_containers, post_translate


class PinToGroundControl(PreProcessControl):
    """Control that moves an object vertically to touch the ground
    Useful when you want a slightly more realistic rendering 
    (i.e. avoid flying objects)

    Continuous Dimensions:

    - ``z_ground``: the z-coordinate that the bottom of the object's bounding
       moves to. Range: ``[0, 1]``.

    .. note::
        This control come after the ``PositionControl`` and
        ``OrientationControl`` controls. In these patterns, you first move the object to a location
        of interest, then drag it to the ground under that location.

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/pin_to_ground/images/image_1.png
            :width: 100
            :group: pin_to_ground

        .. thumbnail:: /_static/logs/pin_to_ground/images/image_2.png
            :width: 100
            :group: pin_to_ground

        .. thumbnail:: /_static/logs/pin_to_ground/images/image_3.png
            :width: 100
            :group: pin_to_ground

        .. thumbnail:: /_static/logs/pin_to_ground/images/image_4.png
            :width: 100
            :group: pin_to_ground

        .. thumbnail:: /_static/logs/pin_to_ground/images/image_5.png
            :width: 100
            :group: pin_to_ground

        Varying ``z_ground`` across its range.
    """

    def __init__(self, root_folder: str):
        continuous_dims = {
            'z_ground': (0., 1.),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Pins the object to the ground

        Parameters
        ----------
        context : Dict[str, Any]
            The scene context object
        control_args : Dict[str, Any]
            The parameters for this control; should have key ``z_ground``
            mapping to any real number, see class docstring for documentation.
        """
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        obj = context["object"]
        bpy.context.view_layer.update()
        obj_min_z = min((obj.matrix_world @ x.co)[2] for x in obj.data.vertices)
        post_translate(obj, Vector([0, 0, control_args['z_ground'] - obj_min_z]))

    def unapply(self, context):
        cleanup_translate_containers(context['object'])

Control = PinToGroundControl
