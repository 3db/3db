import importlib
from collections import defaultdict
from os import path, remove
from glob import glob
from multiprocessing import cpu_count
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
import cv2
import numpy as np

from sandbox.rendering.utils import ControlsApplier

try:
    import bpy  # blender is not required in the master node
except:
    pass

IMAGE_FORMAT = 'png'

def get_model_path(root_folder, model):
    return path.join(root_folder, 'blender_models', model)

def get_env_path(root_folder, env):
    return path.join(root_folder, 'blender_environments', env)

def enumerate_models(root_folder):
    return [path.basename(x) for x in glob(get_model_path(root_folder,
                                                          '*.blend'))]

def enumerate_environments(root_folder):
    return [path.basename(x) for x in glob(get_env_path(root_folder,
                                                        '*.blend'))]

def load_env(root_folder, env):
    bpy.ops.wm.open_mainfile(filepath=get_env_path(root_folder, env))

def load_model(root_folder, model):
    basename, filename = path.split(get_model_path(root_folder, model))
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

    return bpy.data.objects[uid]

def get_model_uid(loaded_model):
    return loaded_model.name

def setup_render(args):
    scene = bpy.context.scene
    bpy.context.scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()  # important to update the list of devices
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.render.tile_x = args.tile_size
    bpy.context.scene.render.tile_y = args.tile_size
    bpy.context.scene.render.film_transparent = True

    for device in cprefs.devices:
        device.use = False

    if args.cpu_cores:
        scene.render.threads_mode = 'FIXED'
        cores_available = cpu_count()
        assert args.cpu_cores <= cores_available, f'Your machine has only {args.cpu_cores} cores.'
        scene.render.threads = max(1, args.cpu_cores)

    if args.gpu_id == -1:
        scene.cycles.device = 'CPU'
        CPU_DEVICES = [x for x in cprefs.devices if x.type == 'CPU']
        CPU_DEVICES[0].use = True
    else:
        scene.cycles.device = 'GPU'
        GPU_DEVICES = [x for x in cprefs.devices if x.type == 'CUDA']
        if len(GPU_DEVICES) != 0:
            GPU_DEVICES[args.gpu_id].use = True
        else:
            raise ValueError('No GPUs found.')
    for d in cprefs.devices:
        print(f'Device {d.name} ({d.type}) used? {d.use}')

    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.render.use_persistent_data = True


def render(uid, job, cli_args, renderer_settings, applier):

    renderer_settings = SimpleNamespace(**vars(cli_args),
                                        **renderer_settings)
    setup_render(renderer_settings)

    context = {
        'object': bpy.context.scene.objects[uid]
    }

    applier.apply_pre_controls(context)

    img_extension = f".{IMAGE_FORMAT}"

    with NamedTemporaryFile(suffix=img_extension) as temp_file:
        temp_filename = temp_file.name
        temp_file.close()
        bpy.context.scene.render.filepath = temp_filename
        bpy.context.scene.render.image_settings.file_format = IMAGE_FORMAT.upper()
        bpy.ops.render.render(use_viewport=False, write_still=True)
        img = cv2.imread(temp_filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        remove(temp_filename) 

    applier.unapply()

    return img
