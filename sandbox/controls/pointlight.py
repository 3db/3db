import numpy as np
from colorsys import hsv_to_rgb
from .base_control import BaseControl

class PointLightControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'H': (0, 1),
        'S': (0, 1),
        'V': (0, 1),
        'intensity': (1000, 10000),
        'distance': (5,20),
        'dir_x': (-1, 1),
        'dir_y': (-1, 1),
        'dir_z': (0, 1),
    }

    discrete_dims = {}

    def apply(self, context, H, S, V, intensity, distance, dir_x, dir_y, dir_z):
        import bpy

        bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD')
        light_object = bpy.context.scene.objects["Point"]
        light = light_object.data
        ob = context['object']

        light_direction = np.array([dir_x, dir_y, dir_z])
        light_direction = light_direction / np.linalg.norm(light_direction)

        # Set light location to point on sphere around object
        light_object.location = distance * light_direction + ob.location

        # Change light properties
        light.type = "POINT"
        light.color = hsv_to_rgb(H, S, V)
        light.energy = intensity

        # Optional - point light towards object
        bpy.ops.object.constraint_add(type="TRACK_TO")
        bpy.context.object.constraints["Track To"].target = ob

        return light_object

Control = PointLightControl

