"""Defines the Blender Camera Control"""

import numpy as np
from threedb.controls.base_control import BaseControl


class CameraControl(BaseControl):
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
    1 - We set the aperture and focal_length
    2 - We move the camera at view_point and look at the object
    3 - We move closer or further in order to satisfy the zoom_factor
        constraint

    As a result in most cases, the final position of the camera will be
    quite different from the view_point parameter.
    """
    kind = 'pre'

    continuous_dims = {
        'view_point_x': (-1, 1),
        'view_point_y': (-1, 1),
        'view_point_z': (0, 1),
        'zoom_factor': (0.5, 2),
        'aperture': (1, 32),
        'focal_length': (10, 400),
    }

    discrete_dims = {}

    def apply(self, context, view_point_x: float, view_point_y: float,
              view_point_z: float, zoom_factor: float, aperture: float,
              focal_length: float):
        """Move and update the camera settings

        Parameters
        ----------
        context
            The scene context
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
        """
        import bpy
        from mathutils import Vector
        from threedb.rendering.utils import (sample_upper_sphere,
                                             lookat_viewport)

        zoomout_factor = 1 / zoom_factor

        camera = bpy.data.objects['Camera']
        camera.data.lens = focal_length
        camera.data.dof.aperture_fstop = aperture
        camera.data.clip_start = 0.001

        ob = context['object']

        # avoid division by zero bug
        view_point_x += 1e-6
        view_point_y += 1e-6
        view_point_z += 1e-6

        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                ctx = bpy.context.copy()
                space = area.spaces[0]
                r = space.region_3d
                ctx['area'] = area
                ctx['region'] = area.regions[-1]
                ctx['space_data'] = space
                ctx['camera'] = camera
                r.view_location = ob.location
                r.view_distance = 2
                previous_lens = space.lens
                space.lens = camera.data.lens * zoomout_factor
                direction = Vector((view_point_x, view_point_y, view_point_z))
                r.view_rotation = lookat_viewport(r.view_location,
                                                  r.view_location + direction)
                bpy.ops.view3d.view_selected(ctx)
                bpy.ops.view3d.camera_to_view(ctx)
                space.lens = previous_lens
                camera.data.dof.focus_object = ob


BlenderControl = CameraControl
