import numpy as np
from colorsys import hsv_to_rgb
from threedb.controls.base_control import BaseControl

class PointLightControl(BaseControl):
    """This control adds a point light in the scene.

    Continuous Parameters
    ---------------------
    H, S, V
        The color of the light
    intensity
        The insity of the light. Value depends on the environment.
    distance
        The distance away from the object of interest
    dir_x
        relative X-coordinate of the point light w.r.t the object of interest
    dir_y
        relative Y-coordinate of the point light w.r.t the object of interest
    dir_z
        relative Z-coordinate of the point light w.r.t the object of interest

    Note
    ----
    1- You can add multiple point lights by using this control multiple times

    2- The vector `(dir_x, dir_y, dir_z)` is used along which direction w.r.t the 
    is the point light placed, but the actual distance along that direction 
    is decided by the `distance` parameter.
    """

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
        """Spawns a point light in the scene pointing to the object of interest.

        Parameters
        ----------
        context
            The scene context            
        H, S, V
            The color of the light
        intensity
            The insity of the light. Value depends on the environment.
        distance
            The distance away from the object of interest
        dir_x
            relative X-coordinate of the point light w.r.t the object of interest
        dir_y
            relative Y-coordinate of the point light w.r.t the object of interest
        dir_z
            relative Z-coordinate of the point light w.r.t the object of interest
        """
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

BlenderControl = PointLightControl
