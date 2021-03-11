import numpy as np
from ..base_control import BaseControl
from .blender_utils import post_translate, cleanup_translate_containers

class ObjLocInFrameControl(BaseControl):
    """Control that moves the object of interest in the image frame.
    This is done by moving the object in a plane parallel to the camera
    plane and passing through the object's position.

    Continuous Dimensions
    ---------------------
    x_shift
        The normalized X-coordinate of the center of the object in the frame.
        Takes any value between -1 (left of the frame) and 1 (right of the frame).
    Y_shift
        The normalized Y-coordinate of the center of the object in the frame
        Takes any value between -1 (bottom of the frame) and 1 (top of the frame).

    Note
    ----
    Setting x_shift and y_shift to zeros would keep the object in the middle of the
    frame.
    """
    kind = 'pre'

    continuous_dims = {
        'x_shift': (-1, 1),
        'y_shift': (-1, 1),
    }

    discrete_dims = {}

    def apply(self, context, x_shift, y_shift):
        """Move the object in the frame

        Parameters
        ----------
        context
            The scene context
        x_shift
            The normalized X-coordinate of the center of the object in the frame.
            Takes any value between -1 (left of the frame) and 1 (right of the frame).
        Y_shift
            The normalized Y-coordinate of the center of the object in the frame
            Takes any value between -1 (bottom of the frame) and 1 (top of the frame).

        """
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

    def unapply(self, context):
        cleanup_translate_containers(self.ob)



BlenderControl = ObjLocInFrameControl
