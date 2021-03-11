"""Defines the Blender Camera Control"""

from typing import Any, Dict

import copy
import bpy
from mathutils import Vector
from ..base_control import PreProcessControl
from ...rendering.utils import lookat_viewport


class CameraControl(PreProcessControl):
    """Control that changes the camera that will be used to render the image

    Continuous Dimensions
    ---------------------
    view_point_x
        The original x coordinate of the camera (see note)
    view_point_y
        The original y coordinate of the camera (see note)
    view_point_z
        The original z coordinate of the camera (see note)
    zoom_factor
        Defines how much should we see of the object. 1 means we completely
        see the object with a little margin. above 1 we are close. below 1
        we are further.
    aperture
        The aperture of the camera
    focal_length
        The focal length of the camera

    Note
    ----
    Since it is impossible to satisfy view_point, zoom and focal length, we
    do the following:

    1. We set the aperture and focal_length

    2. We move the camera at view_point and look at the object

    3. We move closer or further in order to satisfy the zoom_factor
        constraint

    As a result in most cases, the final position of the camera will be
    quite different from the view_point parameter.
    """
    continuous_dims = {
        'view_point_x': (-1, 1),
        'view_point_y': (-1, 1),
        'view_point_z': (0, 1),
        'zoom_factor': (0.5, 2),
        'aperture': (1, 32),
        'focal_length': (10, 400),
    }

    discrete_dims = {}

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Move and update the camera settings.

        Parameters
        ----------
        context : Dict[str, Any]
            The blender scene context.
        control_args : Dict[str, Any]
            How to update the camera. Must have keys:

            - ``view_point_x``: The original x coordinate of the camera.
            - ``view_point_y``: The original y coordinate of the camera.
            - ``view_point_z``: The original z coordinate of the camera.
            - ``zoom_factor``: Defines how much should we see of the object.
                1 means we completely see the object with a little margin.
                Higher value means closer, lower means further away.
            - ``aperture``: The aperture of the camera
            - ``focal_length``: The focal length of the camera
        """
        args_check = self.check_arguments(control_args)
        assert args_check[0], args_check[1]

        args = copy.copy(control_args)
        zoomout_factor = 1 / args['zoom_factor']

        camera = bpy.data.objects['Camera']
        camera.data.lens = args['focal_length']
        camera.data.dof.aperture_fstop = args['aperture']
        camera.data.clip_start = 0.001

        obj = context['object']

        # avoid division by zero bug
        args['view_point_x'] += 1e-6
        args['view_point_y'] += 1e-6
        args['view_point_z'] += 1e-6

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                ctx = bpy.context.copy()
                space = area.spaces[0]
                reg = space.region_3d
                ctx['area'] = area
                ctx['region'] = area.regions[-1]
                ctx['space_data'] = space
                ctx['camera'] = camera
                reg.view_location = obj.location
                reg.view_distance = 2
                previous_lens = space.lens
                space.lens = camera.data.lens * zoomout_factor
                v_dir = Vector((args['view_point_x'], args['view_point_y'], args['view_point_z']))
                reg.view_rotation = lookat_viewport(reg.view_location,
                                                    reg.view_location + v_dir)
                bpy.ops.view3d.view_selected(ctx)
                bpy.ops.view3d.camera_to_view(ctx)
                space.lens = previous_lens
                camera.data.dof.focus_object = obj

BlenderControl = CameraControl
