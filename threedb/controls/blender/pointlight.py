"""
threedb.controls.blender.pointlight
===================================

EXPERIMENTAL CONTROL. DO NOT USE.
"""

from typing import Any, Dict
from ...try_bpy import bpy
import numpy as np
from colorsys import hsv_to_rgb
from threedb.controls.base_control import PreProcessControl

class PointLightControl(PreProcessControl):
    """This control adds a point light in the scene.

    Continuous Parameters:

    - ``H``, ``S``, ``V``: The color of the light (range: ``[0, 1]`` for each)
    - ``intensity``: The intensity of the light. Value depends on the environment.
      (range: ``[1000, 10000]``)
    - ``distance``: The distance away from the object of interest
    - ``dir_x``: relative (x, y, z)-coordinate of the point light w.r.t the object
      of interest. (ranges: ``[-1, 1]``, ``[-1, 1]``, ``[0, 1]`` respectively for x, y, z).

    .. note::
        1. You can add multiple point lights by using this control multiple
        times

        2. The vector ``(dir_x, dir_y, dir_z)`` is used along which direction
        w.r.t the is the point light placed, but the actual distance along
        that direction is decided by the ``distance`` parameter.

    """
    def __init__(self, root_folder: str):
        continuous_dims = {
            'H': (0, 1),
            'S': (0, 1),
            'V': (0, 1),
            'intensity': (1000, 10000),
            'distance': (5, 20),
            'dir_x': (-1, 1),
            'dir_y': (-1, 1),
            'dir_z': (0, 1),
        }
        super().__init__(root_folder, continuous_dims=continuous_dims)

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD')
        light_object = bpy.context.scene.objects["Point"]
        light = light_object.data
        obj = context['object']

        light_direction = np.array([control_args['dir_x'],
                                    control_args['dir_y'],
                                    control_args['dir_z']])
        light_direction = light_direction / np.linalg.norm(light_direction)

        # Set light location to point on sphere around object
        light_object.location = control_args['distance'] * light_direction + obj.location

        # Change light properties
        light.type = "POINT"
        light.color = hsv_to_rgb(control_args['H'],
                                 control_args['S'],
                                 control_args['V'])
        light.energy = control_args['intensity']

        # Optional - point light towards object
        bpy.ops.object.constraint_add(type="TRACK_TO")
        bpy.context.object.constraints["Track To"].target = obj

        return light_object

    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = PointLightControl
