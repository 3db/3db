import numpy as np
from .base_control import BaseControl

class ObjLocInFrameControl(BaseControl):
    kind = 'pre'

    continuous_dims = {
        'x_shift': (-1, 1),
        'y_shift': (-1, 1),
    }

    discrete_dims = {}

    def apply(self, context, x_shift, y_shift):
        from bpy import context as C
        from math import tan
        import numpy as np

        ob = context['object']

        aspect =  C.scene.render.resolution_x/C.scene.render.resolution_y
        camera = C.scene.objects['Camera']
        fov = camera.data.angle_y
        z_obj_wrt_camera = np.linalg.norm(camera.location - ob.location)

        y_limit = tan(fov/2) * z_obj_wrt_camera
        x_limit = y_limit * aspect

        camera_matrix = np.array(C.scene.camera.matrix_world)
        shift = np.matmul(camera_matrix, np.array([[x_limit*x_shift, y_limit*y_shift, -z_obj_wrt_camera, 1]]).T)
        ob.location = shift[:3]


Control = ObjLocInFrameControl

