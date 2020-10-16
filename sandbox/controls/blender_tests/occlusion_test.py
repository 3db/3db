import bpy
from bpy import context as C
from math import tan
import numpy as np

print("Hey")


def move_in_plane(ob, x_shift, y_shift):
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


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.
    Negative 'z' value means the point is behind the camera.
    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.
    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """
    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)
    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != "ORTHO"
    lx = []
    ly = []
    for v in me.vertices:
        co_local = v.co
        z = -co_local.z
        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)
        lx.append(x)
        ly.append(y)
    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)
    mesh_eval.to_mesh_clear()
    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac
    # Sanity check
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)
    return (
        round(min_x * dim_x),  # X
        round(dim_y - max_y * dim_y),  # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y),  # Height
    )


cube = bpy.context.scene.objects["Cube"]
monkey = bpy.context.scene.objects["Suzanne"]

# Print the result
bb = camera_view_bounds_2d(C.scene, C.scene.camera, cube)
bb_occ = camera_view_bounds_2d(C.scene, C.scene.camera, monkey)
print(bb_occ)
print("Hey")


def find_center(bb, bb_occ, dir, occlusion_ratio=0.1):
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


# dir = (0, 1)
dir = (-1, 1)
x_shift, y_shift = find_center(bb, bb_occ, dir, 0.5)
print(f"{x_shift}, {y_shift}")
x_shift = (x_shift - C.scene.render.resolution_x / 2) / (
    C.scene.render.resolution_x / 2
)
y_shift = (y_shift - C.scene.render.resolution_y / 2) / (
    C.scene.render.resolution_y / 2
)
print(f"{x_shift}, {y_shift}")

move_in_plane(monkey, x_shift, y_shift)
# move_in_plane(monkey, x_shift, y_shift)

