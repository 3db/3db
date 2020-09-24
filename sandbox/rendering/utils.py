import numpy as np
import mathutils


def sample_upper_sphere():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    vec[2] = np.abs(vec[2])

    return mathutils.Vector(vec)


def lookat_viewport(target, location):
    diff = location - target
    diff = diff.normalized()
    rot_z = (np.arctan(diff.y / diff.x))
    if diff.x > 0:
        rot_z += np.pi / 2
    else:
        rot_z -= np.pi / 2
    return mathutils.Euler((np.arccos(diff[2]), 0, rot_z)).to_quaternion()
