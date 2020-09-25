import bpy
import importlib
from collections import defaultdict
from os import path
from multiprocessing import cpu_count
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
import cv2

def load_env(env):
    bpy.ops.wm.open_mainfile(filepath=env)

def load_model(model):
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


def setup_render(args):
    scene = bpy.context.scene
    bpy.context.scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices() #important to update the list of devices
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


def render(uid, job, cli_args, renderer_settings):

    control_list = job.control_order
    render_args = job.render_args
    renderer_settings = SimpleNamespace(**vars(cli_args),
                                        **renderer_settings)

    setup_render(renderer_settings)

    control_classes = []

    context = {
        'object': bpy.context.scene.objects[uid]
    }

    for module, classname in control_list:
        imported = importlib.import_module(f'{module}')
        control_classes.append(getattr(imported, classname)())

    groupped_args = defaultdict(dict)

    for (classname, attribute), value in render_args.items():
        groupped_args[classname][attribute] = value

    for control_class in control_classes:
        classname = type(control_class).__name__
        args = groupped_args[type(control_class).__name__]
        control_class.apply(context=context, **args)

    img_extension = f".{renderer_settings.image_format}"

    with NamedTemporaryFile(suffix=img_extension) as temp_file:
        temp_filename = temp_file.name
        temp_file.close()
        print("FNAME", temp_filename)
        bpy.context.scene.render.filepath = temp_filename
        bpy.context.scene.render.image_settings.file_format = renderer_settings.image_format.upper()
        bpy.ops.render.render(use_viewport=False, write_still=True)
        img = cv2.imread(temp_filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

