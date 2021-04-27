"""
threedb.controls.blender.camera
===============================

Control the camera. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/camera.yaml>`_.
"""

from typing import Any, Dict

import copy
from ...try_bpy import bpy
from mathutils import Vector
from ..base_control import PreProcessControl
from ...rendering.utils import lookat_viewport


class CameraControl(PreProcessControl):
    """Control that changes the camera that will be used to render the image

    Continuous Dimensions:

    - ``view_point_x``: The original x coordinate of the camera (see the note
      below). (range: ``[-1, 1]``)
    - ``view_point_y``: The original y coordinate of the camera (see the note
      below). (range: ``[-1, 1]``)
    - ``view_point_z``: The original z coordinate of the camera (see the note
      below). (range: ``[0, 1]``)
    - ``zoom_factor``: Defines how much should we see of the object. A
      ``zoom_factor`` of 1 means we completely see the object with a little
      margin. above 1 we are close. 
      below 1 we are further. (range: ``[0.5, 2]``)
    - ``aperture``: The aperture of the camera. (range: ``[1, 32]``)
    - ``focal_length``: The focal length of the camera. (range: ``[10, 400]``)

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/camera/images/image_1.png
            :width: 100
            :group: camera

        .. thumbnail:: /_static/logs/camera/images/image_2.png
            :width: 100
            :group: camera

        .. thumbnail:: /_static/logs/camera/images/image_3.png
            :width: 100
            :group: camera

        .. thumbnail:: /_static/logs/camera/images/image_4.png
            :width: 100
            :group: camera

        .. thumbnail:: /_static/logs/camera/images/image_5.png
            :width: 100
            :group: camera

        Varying each parameter across its range.

    .. note::
        The camera uses the parameters as following:

        1. We set the ``aperture`` and ``focal_length``
        2. We move the camera according to ``view_point_{x,y,z}`` and look at
           the object
        3. We move closer or further in order to satisfy the ``zoom_factor``
           constraint

        In most cases the final position of the camera will be
        quite different from the ``view_point`` parameter.
    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'view_point_x': (-1, 1),
            'view_point_y': (-1, 1),
            'view_point_z': (0, 1),
            'zoom_factor': (0.5, 2),
            'aperture': (1, 32),
            'focal_length': (10, 400),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

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

    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = CameraControl
