import numpy as np
from sandbox.controls.base_control import BaseControl
try:
    from sandbox.controls.blender.blender_utils import (
        post_translate, cleanup_translate_containers)
except:
    pass

class ObjLocInFrameControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'x_shift': (-1, 1),
        'y_shift': (-1, 1),
    }

    discrete_dims = {}

    def apply(self, context, x_shift, y_shift):
        import bpy
        from bpy import context as C
        from math import tan
        from mathutils import Vector

        ob = context['object']
        self.ob = ob

        bpy.context.view_layer.update()

        aspect = C.scene.render.resolution_x/C.scene.render.resolution_y
        camera = C.scene.objects['Camera']
        fov = camera.data.angle_y
        z_obj_wrt_camera = np.linalg.norm(camera.location - ob.location)

        y_limit = tan(fov/2) * z_obj_wrt_camera
        x_limit = y_limit * aspect

        camera_matrix = np.array(C.scene.camera.matrix_world)
        shift = np.matmul(camera_matrix, np.array([[x_limit*x_shift, y_limit*y_shift, -z_obj_wrt_camera, 1]]).T)
        post_translate(ob, Vector(list(shift[:3])))

    def unapply(self):
        cleanup_translate_containers(self.ob)



BlenderControl = ObjLocInFrameControl
