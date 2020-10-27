import numpy as np
from collections import defaultdict
import importlib


def sample_upper_sphere():
    import mathutils
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    vec[2] = np.abs(vec[2])

    return mathutils.Vector(vec)


def lookat_viewport(target, location):
    import mathutils
    diff = location - target
    diff = diff.normalized()
    rot_z = (np.arctan(diff.y / diff.x))
    if diff.x > 0:
        rot_z += np.pi / 2
    else:
        rot_z -= np.pi / 2
    return mathutils.Euler((np.arccos(diff[2]), 0, rot_z)).to_quaternion()


class ControlsApplier:

    def __init__(self, control_list, render_args, controls_args, root_folder):

        control_classes = []

        for module, classname in control_list:
            imported = importlib.import_module(f'{module}')
            control_classes.append(
                getattr(imported, classname)(
                    root_folder=root_folder,
                    **controls_args[classname])
            )

        grouped_args = defaultdict(dict)

        for (classname, attribute), value in render_args.items():
            grouped_args[classname][attribute] = value

        self.control_classes = control_classes
        self.grouped_args = grouped_args

    def apply_pre_controls(self, context):

        for control_class in self.control_classes:
            if control_class.kind == 'pre':
                classname = type(control_class).__name__
                control_params = self.grouped_args[classname]
                control_class.apply(context=context, **control_params)

    # Unapply controls (e.g. delete occlusion objects, rescale object, etc)
    def unapply(self):
        for control_class in self.control_classes:
            control_class.unapply()

    # post-processing controls
    def apply_post_controls(self, img):
        for control_class in self.control_classes:
            if control_class.kind == 'post':
                classname = type(control_class).__name__
                control_params = self.grouped_args[classname]
                img = control_class.apply(img=img, **control_params)
        return img
