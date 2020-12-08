import importlib
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

def create_node_tree(folder_base, with_uv=False, with_depth=False):
    to_collect = {}
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    scene.view_layers["View Layer"].use_pass_uv = with_uv
    bpy.context.scene.view_layers["View Layer"].use_pass_z = with_depth


    for node in list(nodes):
        nodes.remove(node)

    layers_node = nodes.new(type="CompositorNodeRLayers")

    rgba_image_output_node = nodes.new(type="CompositorNodeOutputFile")
    to_collect['rgb'] = path.join(folder_base, "rgb")
    rgba_image_output_node.base_path = to_collect['rgb']

    links.new(layers_node.outputs[0], rgba_image_output_node.inputs[0])

    if with_uv:
        uv_image_output_node = nodes.new(type="CompositorNodeOutputFile")
        links.new(layers_node.outputs["UV"], uv_image_output_node.inputs[0])
        to_collect['uv'] = path.join(folder_base, 'uv')
        uv_image_output_node.base_path = to_collect['uv']

    if with_depth:
        depth_image_output_node = nodes.new(type="CompositorNodeOutputFile")
        links.new(layers_node.outputs["Depth"], depth_image_output_node.inputs[0])
        depth_image_output_node.base_path = folder_base + "_depth"
        to_collect['depth'] = path.join(folder_base, 'depth')
        depth_image_output_node.base_path = to_collect['depth']

    return to_collect


def setup_render(args):
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

def render(uid, job, cli_args, renderer_settings, applier,
           loaded_model=None, loaded_env=None):

    context = {
        'object': bpy.context.scene.objects[uid]
    }

    applier.apply_pre_controls(context)

    output = {}

    with TemporaryDirectory() as temp_folder:
        print(temp_folder)
        to_collect = create_node_tree(temp_folder,
                                      renderer_settings['with_uv'],
                                      renderer_settings['with_depth'])
        bpy.context.scene.render.image_settings.file_format = IMAGE_FORMAT.upper()
        bpy.ops.render.render(use_viewport=False, write_still=True)
        for name, folder_to_collect in to_collect.items():
            images = glob(path.join(folder_to_collect, '*'))
            assert len(images) == 1
            temp_filename = images[0]
            img = cv2.imread(temp_filename, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            output[name] = ch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        # temp_folder.cleanup()

    return output
