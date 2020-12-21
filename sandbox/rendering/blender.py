import importlib
import re
import torch as ch
from collections import defaultdict
from os import path, remove
from glob import glob
from multiprocessing import cpu_count
from tempfile import TemporaryDirectory
import cv2
import numpy as np

from sandbox.rendering.utils import ControlsApplier

try:
    import bpy  # blender is not required in the master node
except:
    pass

IMAGE_FORMAT = 'png'

ENV_EXTENSIONS = ['blend', 'exr', 'hdr']

NAME = "Blender"

def get_model_path(root_folder, model):
    return path.join(root_folder, 'blender_models', model)

def get_env_path(root_folder, env):
    return path.join(root_folder, 'blender_environments', env)

def enumerate_models(root_folder):
    return [path.basename(x) for x in glob(get_model_path(root_folder,
                                                          '*.blend'))]

def enumerate_environments(root_folder):
    all_files = [path.basename(x) for x in
                 glob(get_env_path(root_folder, '*.*'))]
    all_files = [x for x in all_files if x.split('.')[-1] in ENV_EXTENSIONS]
    return all_files

def load_env(root_folder, env):
    full_env_path = get_env_path(root_folder, env)

    if env.endswith('.blend'): # full blender file
        bpy.ops.wm.open_mainfile(filepath=full_env_path)
    else:  # HDRI env
        bpy.ops.wm.read_homefile()
        bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
        bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)
        bpy.context.scene.render.film_transparent = False
        world = bpy.context.scene.world
        node_tree = world.node_tree
        output_node = world.node_tree.get_output_node('CYCLES')

        [node_tree.links.remove(x) for x in output_node.inputs[0].links]

        background_node = node_tree.nodes.new(type="ShaderNodeBackground")
        node_tree.links.new(background_node.outputs[0], output_node.inputs[0])

        img = bpy.data.images.load(full_env_path)
        env_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        env_texture_node.image = img

        node_tree.links.new(env_texture_node.outputs[0], background_node.inputs[0])


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

def setup_nice_PNG(input_node):
    input_node.use_node_format = False
    input_node.format.file_format = "PNG"
    input_node.format.compression = 0
    input_node.format.color_depth = "16"

MAIN_NODES = []
POST_PROCESS_NODES = []

def before_render():
    # COLOR settings for render
    bpy.context.scene.display_settings.display_device = 'None'
    bpy.context.scene.sequencer_colorspace_settings.name = 'Raw'
    bpy.context.view_layer.update()
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'

    for node in MAIN_NODES:
        node.mute = False

    for node in POST_PROCESS_NODES:
        node.mute = True

def before_preprocess():
    # COLOR SETTINGS for RGB output
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'
    bpy.context.view_layer.update()
    bpy.context.scene.view_settings.view_transform = 'Filmic'
    bpy.context.scene.view_settings.look = 'None'

    for node in MAIN_NODES:
        node.mute = True

    for node in POST_PROCESS_NODES:
        node.mute = False


def setup_render(args):
    while MAIN_NODES:
        MAIN_NODES.pop()
    while POST_PROCESS_NODES:
        POST_PROCESS_NODES.pop()

    scene = bpy.context.scene
    bpy.context.scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()  # important to update the list of devices
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.render.tile_x = args.tile_size
    bpy.context.scene.render.tile_y = args.tile_size

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

    scene.use_nodes = True

    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    scene.view_layers["View Layer"].use_pass_uv = args.with_uv
    bpy.context.scene.view_layers["View Layer"].use_pass_z = args.with_depth
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = args.with_segmentation

    scene.use_nodes = True
    scene.name = 'main_scene'

    for node in list(nodes):
        nodes.remove(node)

    layers_node = nodes.new(type="CompositorNodeRLayers")
    MAIN_NODES.append(layers_node)

    file_output_node = nodes.new(type="CompositorNodeOutputFile")
    file_output_node.name = 'exr_output'
    MAIN_NODES.append(file_output_node)
    file_output_node.format.file_format = "OPEN_EXR"
    file_output_node.format.exr_codec = 'NONE'
    output_slots = file_output_node.file_slots
    output_slots.remove(file_output_node.inputs[0])
    output_slots.new("render_exr")
    links.new(layers_node.outputs[0], file_output_node.inputs[0])

    if args.with_depth:
        output_slots.new("depth")
        setup_nice_PNG(file_output_node.file_slots["depth"])
        math_node = nodes.new(type="CompositorNodeMath")
        MAIN_NODES.append(math_node)
        links.new(layers_node.outputs["Depth"], math_node.inputs[0])
        math_node.operation = "DIVIDE"
        math_node.inputs[1].default_value = args.max_depth
        links.new(math_node.outputs[0], file_output_node.inputs["depth"])

    if args.with_uv:
        output_slots.new("uv")
        setup_nice_PNG(file_output_node.file_slots["uv"])
        links.new(layers_node.outputs["UV"], file_output_node.inputs["uv"])

    if args.with_segmentation:
        output_slots.new("segmentation")
        setup_nice_PNG(file_output_node.file_slots["segmentation"])
        file_output_node.file_slots["segmentation"].format.color_mode = "BW"
        math_node = nodes.new(type="CompositorNodeMath")
        MAIN_NODES.append(math_node)
        links.new(layers_node.outputs["IndexOB"], math_node.inputs[0])
        math_node.operation = "DIVIDE"
        math_node.inputs[1].default_value = 65535.0
        links.new(math_node.outputs[0], file_output_node.inputs["segmentation"])

    input_image = nodes.new(type="CompositorNodeImage")
    POST_PROCESS_NODES.append(input_image)
    input_image.name = "input_image"
    file_output_node = nodes.new(type="CompositorNodeOutputFile")
    file_output_node.name = "rgb_output"
    POST_PROCESS_NODES.append(file_output_node)
    output_slots = file_output_node.file_slots
    output_slots.remove(file_output_node.inputs[0])
    output_slots.new("rgb")
    file_output_node.format.file_format = "PNG"
    file_output_node.format.compression = 0
    file_output_node.format.color_depth = "8"
    links.new(input_image.outputs["Image"], file_output_node.inputs["rgb"])

count = 0

def render(uid, object_class, job, cli_args, renderer_settings, applier,
           loaded_model=None, loaded_env=None):
    global count

    context = {
        'object': bpy.context.scene.objects[uid]
    }

    # 0 is background so we shift everything by 1
    context['object'].pass_index = object_class + 1

    applier.apply_pre_controls(context)

    output = {}

    with TemporaryDirectory() as temp_folder:
        scene = bpy.context.scene
        scene.node_tree.nodes['exr_output'].base_path = temp_folder
        before_render()
        bpy.ops.render.render(use_viewport=False, write_still=False)
        before_preprocess()
        written_file = glob(path.join(temp_folder, '*.exr'))
        blender_loaded_image = bpy.data.images.load(written_file[0])
        scene.node_tree.nodes["input_image"].image = blender_loaded_image
        scene.node_tree.nodes["rgb_output"].format.file_format = IMAGE_FORMAT.upper()
        scene.node_tree.nodes['rgb_output'].base_path = temp_folder
        bpy.ops.render.render(use_viewport=False, write_still=False)

        all_files = glob(path.join(temp_folder, "*.png"))

        for full_filename in all_files:
            name = re.sub(r'[0-9]+.png', '', path.basename(full_filename))
            img = cv2.imread(full_filename, cv2.IMREAD_UNCHANGED)

            if name == 'segmentation': 
                img = img[:, :, None]  # Add extra dimension for the channel
                img = img.astype('int32') - 1  # Go back from the 1 index to the 0 index
                # We needed 1 index for the classes because we can only read images with
                # positive integers
            elif img.dtype is np.dtype(np.uint16):
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                img = img.astype('float32') / (2**16 - 1)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                img = img.astype('float32') / (2**8 - 1)

            output[name] = ch.from_numpy(img).permute(2, 0, 1)

        # Avoid memory leak by keeping all EXR rendered so far in memory
        bpy.data.images.remove(blender_loaded_image)
        if count == 10:
            bpy.ops.wm.save_as_mainfile(filepath='/tmp/scene.blend')
            exit()
        else:
            count += 1

    return output
