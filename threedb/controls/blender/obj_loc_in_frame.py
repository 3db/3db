"""
threedb.controls.blender.obj_loc_in_frame
=========================================

Change the object location in the frame. 
An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/obj_loc_in_frame.yaml>`_. 
"""

from ...try_bpy import bpy
# from bpy import context as C
from math import tan
from mathutils import Vector
from typing import Any, Dict
import numpy as np
from ..base_control import BaseControl, PreProcessControl
from .utils import post_translate, cleanup_translate_containers

class ObjLocInFrameControl(PreProcessControl):
    """Control that moves the object of interest in the image frame.
    This is done by moving the object in a plane parallel to the camera
    plane and passing through the object's position.

    Continuous Dimensions:

    - ``x_shift``: The normalized X-coordinate of the center of the object in the frame.
        A value of -1 is the left-most edge of the frame and 1 is the right-most
        edge of the frame. (range: ``[-1, 1]``)
    - ``y_shift``: The normalized Y-coordinate of the center of the object in the frame
        Takes any value between -1 (bottom of the frame) and 1 (top of the frame).

    .. note::
        Setting x_shift and y_shift to zeros keeps the object in the
        middle of the frame.
    
    .. admonition:: Example images

        .. thumbnail:: /_static/logs/obj_loc_in_frame/images/image_1.png
            :width: 100
            :group: obj_loc_in_frame

        .. thumbnail:: /_static/logs/obj_loc_in_frame/images/image_2.png
            :width: 100
            :group: obj_loc_in_frame

        .. thumbnail:: /_static/logs/obj_loc_in_frame/images/image_3.png
            :width: 100
            :group: obj_loc_in_frame

        .. thumbnail:: /_static/logs/obj_loc_in_frame/images/image_4.png
            :width: 100
            :group: obj_loc_in_frame

        .. thumbnail:: /_static/logs/obj_loc_in_frame/images/image_5.png
            :width: 100
            :group: obj_loc_in_frame

        Example renderings while varying the parameters.
    
    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'x_shift': (-1., 1.),
            'y_shift': (-1., 1.),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Move the object in the frame

        Parameters
        ----------
        context
            The scene context.
        x_shift
            The normalized X-coordinate of the center of the object in the frame.
            Takes any value between -1 (left of the frame) and 1 (right of the frame).
        Y_shift
            The normalized Y-coordinate of the center of the object in the frame
            Takes any value between -1 (bottom of the frame) and 1 (top of the frame).

        """
        obj = context['object']
        bpy.context.view_layer.update()

        aspect = bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y
        camera = bpy.context.scene.objects['Camera']
        fov = camera.data.angle_y
        z_obj_wrt_camera = np.linalg.norm(camera.location - obj.location)

        y_limit = tan(fov/2) * z_obj_wrt_camera
        x_limit = y_limit * aspect

        camera_matrix = np.array(bpy.context.scene.camera.matrix_world)
        coords = [x_limit * control_args['x_shift'],
                  y_limit * control_args['y_shift'],
                  - z_obj_wrt_camera, 1]
        shift = np.matmul(camera_matrix, np.array([coords]).T)
        post_translate(obj, Vector(list(shift[:3])))

    def unapply(self, context):
        cleanup_translate_containers(context['object'])

Control = ObjLocInFrameControl
