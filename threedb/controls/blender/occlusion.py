"""
threedb.controls.blender.occlusion
==================================

Cover the subject object with another object. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/occlusion.yaml>`_. 
"""

from ...try_bpy import bpy
from math import tan
from typing import Any, Dict
import numpy as np
from pathlib import Path
from ..base_control import PreProcessControl
from . import utils

class OcclusionControl(PreProcessControl):
    """Control that adds an occlusion object infront of the
    main object in the scene.

    Continuous Dimensions:

    - ``occlusion_ratio``: Ratio of the occluded part of the object of interest.
      e.g., parameter 0.01 would occlude 1% of the object, and parameter 0.8 would occlude 80%
      of the object. Default range: ``(0.01, .8)``.
    - ``zoom``: Controls distance from the occluder to the object of interest. The occluder is
      placed between the object of interest and the camera at a distance of ``zoom * D``,
      measured from the object of interest. ``D`` is the distance between the object of interest
      and the camera. Default range: ``(0.01, 0.4)``.
    - ``scale``: rescale the occlusion object by a factor of ``scale``. Parameter 0.5 halves the
      occluder's size, and parameter 2 doubles the occluder's size. Default range: ``(0.01, 1)``.

    Discrete Dimensions:

    - ``direction``: The direction from which the occluder approaches the
      object of interest. Takes a value between 0 and 7
      represeting the indices of the ``DIRECTIONS`` vectors.
    - ``occluder``: The occlusion object. This is an index of the list of the
      occlusion objects which is automatically initialized when the
      OcclusionControl is created (see Note).

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/occlusion/images/image_1.png
            :width: 100
            :group: occlusion

        .. thumbnail:: /_static/logs/occlusion/images/image_2.png
            :width: 100
            :group: occlusion

        .. thumbnail:: /_static/logs/occlusion/images/image_3.png
            :width: 100
            :group: occlusion

        .. thumbnail:: /_static/logs/occlusion/images/image_4.png
            :width: 100
            :group: occlusion

        .. thumbnail:: /_static/logs/occlusion/images/image_5.png
            :width: 100
            :group: occlusion

        Varying all the parameters.
    """
    DIRECTIONS = [( 1, -1), ( 1, 0), ( 1, 1),
                  ( 0, -1),          ( 0, 1),
                  (-1, -1), (-1, 0), (-1, 1)]
    OOD_DIR = 'ood_objects'

    def __init__(self, root_folder):
        """Initializes the `OcclusionControl`

        Parameters
        ----------
        root_folder
            The root folder where the `ood_objects` folder containing
            the possible occluders exist
        """
        continuous_dims = {
            "occlusion_ratio": (0.01, 0.8),
            "zoom": (0.01, 0.4),
            "scale": (0.01, 1),
        }

        ood_folder = Path(root_folder) / self.OOD_DIR
        occluders_paths = list(ood_folder.glob('*.blend'))
        occluders_paths = [f.name for f in occluders_paths]

        print('ood_folder', ood_folder)
        print('occluders_paths', occluders_paths)

        discrete_dims = {
            "direction": list(range(len(self.DIRECTIONS))),
            "occluder": occluders_paths
        }
        assert len(discrete_dims['occluder']) >= 1, 'No occluder objects found!'
        super().__init__(root_folder,
                         continuous_dims=continuous_dims,
                         discrete_dims=discrete_dims)
        self.occluder_paths = occluders_paths
        self.ood_folder = ood_folder

    def _move_in_plane(self, ob, x_shift, y_shift):
        """Shifts the `ob` object in a plane passing through the
        ob location, and parallel to the camera frame.

        Parameters
        ----------
        ob
            object of interest
        x_shift
            Shift along the X-axis in the camera frame.
            Takes a value between -1 and 1.
        Y_shift
            Shift along the Y-axis in the camera frame
            Takes a value between -1 and 1.
        """
        C = bpy.context

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

    def _find_center(self, bb, bb_occ, dir, occlusion_ratio=0.1):
        """Finds the location of the occlusion object `bb_occ` so that
        it occludes a specific ratio of the object `bb` from a given
        direction

        Parameters
        ----------
        bb
            The occluded object
        bb_occ
            The occluder
        dir
            The direction of occlusion
        occlustion_raio
            The ratio of occlusion
        """
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

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        C = bpy.context
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg       

        ob = context["object"]
        occ_basename = control_args['occluder']
        model_path = self.ood_folder / occ_basename
        occluder = utils.load_model(model_path)
        self.occluder = C.scene.objects[occluder]
        self.occluder.location = ob.location + float(control_args['zoom']) * (C.scene.camera.location - ob.location)

        bb = utils.camera_view_bounds_2d(C.scene, C.scene.camera, ob)
        bb_occ = utils.camera_view_bounds_2d(C.scene, C.scene.camera, self.occluder)
        x_shift, y_shift = self._find_center(bb, bb_occ,
                                             self.DIRECTIONS[control_args['direction']],
                                             control_args['occlusion_ratio'])
        x_shift = (x_shift - C.scene.render.resolution_x / 2) / (
            C.scene.render.resolution_x / 2
        )
        y_shift = (y_shift - C.scene.render.resolution_y / 2) / (
            C.scene.render.resolution_y / 2
        )
        self._move_in_plane(self.occluder, x_shift, y_shift)
        self.occluder.scale = (control_args['scale'],) * 3

    def unapply(self, context: Dict[str, Any]) -> None:
        """Deletes the occlusion objects. This is important to avoid
        clutering the scene with unneeded objects at the subsequent frames.

        Parameters
        ----------
        context : Dict[str, Any]
            The scene context object
        """
        bpy.ops.object.delete({"selected_objects": [self.occluder]})

Control = OcclusionControl
