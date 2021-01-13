import numpy as np
from os import path
from glob import glob
from sandbox.controls.base_control import BaseControl

class OcclusionControl(BaseControl):
    kind = "pre"

    continuous_dims = {
        "occlusion_ratio": (0.1, 1.0),
        "zoom": (0.1, 0.4),
        "scale": (0.25, 1),
    }

    discrete_dims = {
        "direction": 8,
        "occluder": None
    }

    DIRECTIONS = [( 1, -1), ( 1, 0), ( 1, 1),
                  ( 0, -1),          ( 0, 1),
                  (-1, -1), (-1, 0), (-1, 1)]

    def __init__(self, root_folder):
        super().__init__(root_folder)
        self.occluders_paths = [x for x in glob(path.join(root_folder, 'ood_objects', '*.blend'))]
        self.discrete_dims['occluder'] = len(self.occluders_paths)
        assert self.discrete_dims['occluder'] >= 1

    def move_in_plane(self, ob, x_shift, y_shift):
        from bpy import context as C
        from math import tan

        aspect = C.scene.render.resolution_x / C.scene.render.resolution_y

        camera = C.scene.objects["Camera"]
        fov = camera.data.angle_y
        z_obj_wrt_camera = np.linalg.norm(camera.location - ob.location)
        y_limit = tan(fov / 2) * z_obj_wrt_camera
        x_limit = y_limit * aspect
        camera_matrix = np.array(C.scene.camera.matrix_world)
        shift = np.matmul(
            camera_matrix,
            np.array([[x_limit * x_shift, y_limit * y_shift, -z_obj_wrt_camera, 1]]).T,
        )
        ob.location = shift[:3]

    def find_center(self, bb, bb_occ, dir, occlusion_ratio=0.1):
        bb_area = bb[2] * bb[3]
        bb_occ_area = bb_occ[2] * bb[3]
        overlap_area = min(bb_occ_area, bb_area * occlusion_ratio)

        bb_xc = bb[0] + bb[2] / 2
        bb_yc = bb[1] + bb[3] / 2

        aspect = bb[3] / bb[2]

        if dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            penetration = (
                np.abs(dir[0]) * overlap_area / bb_occ[3]
                + np.abs(dir[1]) * overlap_area / bb_occ[2]
            )
            x_out = bb_xc + dir[0] * (bb_occ[2] / 2 + bb[2] / 2 - penetration)
            y_out = bb_yc + dir[1] * (bb_occ[3] / 2 + bb[3] / 2 - penetration)

        elif dir == (1, 1) or (-1, 1) or (1, -1) or (-1, -1):
            penetration_x = np.sqrt(overlap_area * aspect)
            penetration_y = overlap_area / penetration_x
            # assert penetration_x / penetration_y == aspect
            x_out = bb_xc + dir[0] * (bb_occ[2] / 2 + bb[2] / 2 - penetration_x)
            y_out = bb_yc + dir[1] * (bb_occ[3] / 2 + bb[3] / 2 - penetration_y)

        return (x_out, y_out)

    def apply(self, context, direction, zoom, occlusion_ratio, occluder, scale):

        from .blender_utils import camera_view_bounds_2d, load_model
        import mathutils
        from bpy import context as C

        ob = context["object"]
        occluder = load_model(self.occluders_paths[occluder])
        self.occluder = C.scene.objects[occluder]
        self.occluder.location = ob.location + zoom * (C.scene.camera.location - ob.location)

        bb = camera_view_bounds_2d(C.scene, C.scene.camera, ob)
        bb_occ = camera_view_bounds_2d(C.scene, C.scene.camera, self.occluder)
        x_shift, y_shift = self.find_center(bb, bb_occ, self.DIRECTIONS[direction], occlusion_ratio)
        x_shift = (x_shift - C.scene.render.resolution_x / 2) / (
            C.scene.render.resolution_x / 2
        )
        y_shift = (y_shift - C.scene.render.resolution_y / 2) / (
            C.scene.render.resolution_y / 2
        )
        self.move_in_plane(self.occluder, x_shift, y_shift)
        self.occluder.scale = (scale,) * 3

    def unapply(self, context):
        import bpy
        bpy.ops.object.delete({"selected_objects": [self.occluder]})

BlenderControl = OcclusionControl
