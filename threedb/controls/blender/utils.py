"""
Rendering utils
===============

Common utils function for blender controls
"""
from os import path
from typing import Any, Tuple
from ...try_bpy import bpy

def load_model(model: str) -> str:
    """Load an object from a blender file and insert it in the scene

    Note
    ----
    Since .blend files can contain many object we go under the assumption
    that in the file `XXXXXX.blend`, there will be an object named `XXXXXX`,
    and this is this one we will load

    Parameters
    ----------
    model
        The path to the blender file that needs to be loaded

    Returns
    -------
    str
        The uid of the object loaded
    """
    basename, filename = path.split(model)
    uid = filename.replace('.blend', '')
    blendfile = path.join(basename, uid + '.blend')
    section = "\\Object\\"
    object = uid

    filepath = uid + '.blend'
    directory = blendfile + section
    filename = object

    bpy.ops.wm.append(
        filepath=filepath,
        filename=filename,
        directory=directory)

    return uid


TRANSLATE_PREFIX = 'translation_control_'


def cleanup_translate_containers(obj: Any) -> None:
    """Remove all translations containers

    Note
    ----
    To know what translate containers are please refer to `post_translate`

    Parameters
    ----------
    obj: blender object
        The object to remove containers on

    """
    obj.parent = None
    for other in bpy.data.objects:
        if other.type == 'EMPTY' and len(other.children) == 0:
            bpy.data.objects.remove(other, do_unlink=True)


def post_translate(obj: Any, offset: Tuple[float, float, float]) -> None:
    """Apply a translation on an object but ensure it happens after rotations

    Note
    ----
    To work this function creates a container around the object and apply
    the translation on the parent. This way, any rotation on the object
    itself will always be applied before the translation
    The container name will start by TRANSLATE_PREFIX so that is can easily
    be detected and removed in `cleanup_translate_containers`

    Parameters
    ----------
    obj: blender object
        The object to translate
    offset
        The vector to translate by
    """
    if (obj.parent is None
            or not obj.parent.name.startswith(TRANSLATE_PREFIX)):
        bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD',
                                 location=(0, 0, 0), scale=(1, 1, 1))
        container = bpy.context.object
        container.name = TRANSLATE_PREFIX + obj.name
        obj.parent = container
    obj.parent.location += offset


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between two numbers

    Parameters
    ----------
    value
        The value to clamp
    minimum
        Minimum acceptable value
    maximum
        Maximum acceptable value

    Returns
    -------
    float
        The clamped value
    """

    return max(minimum, min(value, maximum))


def camera_view_bounds_2d(scene: bpy.types.Scene,
                          cam_ob: bpy.types.Object,
                          mesh: bpy.types.Mesh) -> Tuple[float, float, float, float]:
    """Returns camera-space bounding box of mesh object.
    Negative 'z' value means the point is behind the camera.
    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    Parameters
    ----------
    scene : bpy.types.Scene
        Scene to use for frame size.
    cam_ob : bpy.types.Object
        Camera object
    me_ob : bpy.types.Mesh
        Untransformed Mesh.

    Returns
    -------
    Tuple[float, float, float, float]
        A tuple ``(x, y, width, height)`` encoding the camera-space bounding box
        of the mesh object.
    """    
    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = mesh.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(mesh.matrix_world)
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

    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)
    return (
        round(min_x * dim_x),  # X
        round(dim_y - max_y * dim_y),  # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y),  # Height
    )
