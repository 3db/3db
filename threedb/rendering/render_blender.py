"""
threedb.rendering.render_blender
================================

Implements the `Blender` rendering class that subclasses :class:`threedb.rendering.base_renderer.BaseRenderer`:.
This includes all the Blender-specific rendering settings, functions, and configs.
"""

import re
from glob import glob
from multiprocessing import cpu_count
from os import path
from tempfile import TemporaryDirectory
from typing import Tuple, Dict, Optional, Iterable, List, Any

from ..try_bpy import bpy

import cv2
import numpy as np
import torch as ch
from .base_renderer import BaseRenderer, RenderEnv, RenderObject
from .utils import ControlsApplier

IMAGE_FORMAT = 'png'

ENV_EXTENSIONS = ['blend', 'exr', 'hdr']

"""
Utility functions
"""
def _get_model_path(root_folder: str, model: str) -> str:
    return path.join(root_folder, 'blender_models', model)

def _get_env_path(root_folder: str, env: str) -> str:
    return path.join(root_folder, 'blender_environments', env)

def _setup_nice_PNG(input_node: Any) -> None:
    input_node.use_node_format = False
    input_node.format.file_format = "PNG"
    input_node.format.compression = 0
    input_node.format.color_depth = "16"

class Blender(BaseRenderer):
    NAME: str = 'Blender'
    KEYS: List[str] = ['rgb', 'segmentation', 'uv', 'depth']

    def __init__(self, root_dir: str, render_settings: Dict[str, Any], _ = None) -> None:
        super().__init__(root_dir, render_settings, ENV_EXTENSIONS)
        self.main_nodes = []
        self.post_process_nodes = []

    @staticmethod
    def enumerate_models(search_dir: str) -> List[str]:
        """
        Given a root folder, returns all .blend files in root/blender_models/
        """
        return [path.basename(x) for x in glob(_get_model_path(search_dir, '*.blend'))]

    @staticmethod
    def enumerate_environments(search_dir: str) -> List[str]:
        all_files = map(lambda x: path.basename(x), glob(_get_env_path(search_dir, '*.*')))
        return list(filter(lambda x: x.split('.')[-1] in ENV_EXTENSIONS, all_files))

    def declare_outputs(self) -> Dict[str, Tuple[List[int], str]]:
        imsize = [self.args['resolution'], self.args['resolution']]
        output_channels: Dict[str, Tuple[List[int], str]] = {'rgb': ([3, *imsize], 'float32')}
        if self.args['with_uv']:
            output_channels['uv'] = ([4, *imsize], 'float32')
        if self.args['with_depth']:
            output_channels['depth'] = ([4, *imsize], 'float32')
        if self.args['with_segmentation']:
            output_channels['segmentation'] = ([1, *imsize], 'int32')

        return output_channels

    def load_model(self, model: str) -> RenderObject:
        basename, filename = path.split(_get_model_path(self.root_dir, model))
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

    def get_model_uid(self, loaded_model):
        return loaded_model.name

    def load_env(self, env: str) -> Optional[RenderEnv]:
        full_env_path = _get_env_path(self.root_dir, env)

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

    def _before_render(self) -> None:
        """
        Private utility function to be called from render()
        """
        # COLOR settings for render
        bpy.context.scene.display_settings.display_device = 'None'
        bpy.context.scene.sequencer_colorspace_settings.name = 'Raw'
        bpy.context.view_layer.update()
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.look = 'None'

        for node in self.main_nodes:
            node.mute = False

        for node in self.post_process_nodes:
            node.mute = True

    def _before_preprocess(self) -> None:
        # COLOR SETTINGS for RGB output
        bpy.context.scene.display_settings.display_device = 'sRGB'
        bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'
        bpy.context.view_layer.update()
        bpy.context.scene.view_settings.view_transform = 'Filmic'
        bpy.context.scene.view_settings.look = 'None'

        for node in self.main_nodes:
            node.mute = True

        for node in self.post_process_nodes:
            node.mute = False
    
    def _setup_render_device(self, scene: Any, prefs: Any):
        gpu_id: int = self.args['gpu_id']
        cpu_cores: Optional[int] = self.args['cpu_cores']

        cprefs = prefs.addons['cycles'].preferences
        cprefs.get_devices()  # important to update the list of devices

        for device in cprefs.devices:
            device.use = False

        if cpu_cores:
            scene.render.threads_mode = 'FIXED'
            cores_available = cpu_count()
            assert cpu_cores <= cores_available, f'Your machine has only {cpu_cores} cores.'
            scene.render.threads = max(1, cpu_cores)

        if gpu_id == -1:
            scene.cycles.device = 'CPU'
            cpu_devices = [x for x in cprefs.devices if x.type == 'CPU']
            cpu_devices[0].use = True
        else:
            scene.cycles.device = 'GPU'
            gpu_devices = [x for x in cprefs.devices if x.type == 'CUDA']
            if len(gpu_devices) != 0:
                gpu_devices[gpu_id].use = True
            else:
                raise ValueError('No GPUs found.')
        for d in cprefs.devices:
            print(f'Device {d.name} ({d.type}) used? {d.use}')

    def setup_render(self, model: Optional[RenderObject], env: Optional[RenderEnv]) -> None:
        while self.main_nodes:
            self.main_nodes.pop()
        while self.post_process_nodes:
            self.post_process_nodes.pop()

        scene = bpy.context.scene
        bpy.context.scene.render.engine = 'CYCLES'
        prefs = bpy.context.preferences

        self._setup_render_device(scene, prefs)

        bpy.context.scene.cycles.samples = self.args['samples']
        bpy.context.scene.render.tile_x = self.args['tile_size']
        bpy.context.scene.render.tile_y = self.args['tile_size']
        bpy.context.scene.render.resolution_x = self.args['resolution']
        bpy.context.scene.render.resolution_y = self.args['resolution']
        bpy.context.scene.render.use_persistent_data = True

        scene.use_nodes = True

        nodes = scene.node_tree.nodes
        links = scene.node_tree.links

        scene.view_layers["View Layer"].use_pass_uv = self.args['with_uv']
        bpy.context.scene.view_layers["View Layer"].use_pass_z = self.args['with_depth']
        bpy.context.scene.view_layers["View Layer"].use_pass_object_index = self.args['with_segmentation']

        scene.use_nodes = True
        scene.name = 'main_scene'

        for node in list(nodes):
            nodes.remove(node)

        layers_node = nodes.new(type="CompositorNodeRLayers")
        self.main_nodes.append(layers_node)

        file_output_node = nodes.new(type="CompositorNodeOutputFile")
        file_output_node.name = 'exr_output'
        self.main_nodes.append(file_output_node)
        file_output_node.format.file_format = "OPEN_EXR"
        file_output_node.format.exr_codec = 'NONE'
        output_slots = file_output_node.file_slots
        output_slots.remove(file_output_node.inputs[0])
        output_slots.new("render_exr")
        links.new(layers_node.outputs[0], file_output_node.inputs[0])

        if self.args['with_depth']:
            output_slots.new("depth")
            _setup_nice_PNG(file_output_node.file_slots["depth"])
            math_node = nodes.new(type="CompositorNodeMath")
            self.main_nodes.append(math_node)
            links.new(layers_node.outputs["Depth"], math_node.inputs[0])
            math_node.operation = "DIVIDE"
            math_node.inputs[1].default_value = self.args['max_depth']
            links.new(math_node.outputs[0], file_output_node.inputs["depth"])

        if self.args['with_uv']:
            output_slots.new("uv")
            _setup_nice_PNG(file_output_node.file_slots["uv"])
            links.new(layers_node.outputs["UV"], file_output_node.inputs["uv"])

        if self.args['with_segmentation']:
            output_slots.new("segmentation")
            _setup_nice_PNG(file_output_node.file_slots["segmentation"])
            file_output_node.file_slots["segmentation"].format.color_mode = "BW"
            math_node = nodes.new(type="CompositorNodeMath")
            self.main_nodes.append(math_node)
            links.new(layers_node.outputs["IndexOB"], math_node.inputs[0])
            math_node.operation = "DIVIDE"
            math_node.inputs[1].default_value = 65535.0
            links.new(math_node.outputs[0], file_output_node.inputs["segmentation"])

        input_image = nodes.new(type="CompositorNodeImage")
        self.post_process_nodes.append(input_image)
        input_image.name = "input_image"
        file_output_node = nodes.new(type="CompositorNodeOutputFile")
        file_output_node.name = "rgb_output"
        self.post_process_nodes.append(file_output_node)
        output_slots = file_output_node.file_slots
        output_slots.remove(file_output_node.inputs[0])
        output_slots.new("rgb")
        file_output_node.format.file_format = "PNG"
        file_output_node.format.compression = 0
        file_output_node.format.color_depth = "8"
        links.new(input_image.outputs["Image"], file_output_node.inputs["rgb"])
    
    def get_context_dict(self, model_uid: str, object_class: int) -> Dict[str, Any]:
        obj  = bpy.context.scene.objects[model_uid]

        # 0 is background so we shift everything by 1
        obj.pass_index = object_class + 1

        return {'object': obj}


    def render(self,
               model_uid: str,
               loaded_model: RenderObject, 
               loaded_env: RenderEnv) -> Dict[str, ch.Tensor]:
        output = {}

        with TemporaryDirectory() as temp_folder:
            scene = bpy.context.scene
            scene.node_tree.nodes['exr_output'].base_path = temp_folder
            self._before_render()
            bpy.ops.render.render(use_viewport=False, write_still=False)
            self._before_preprocess()
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

        return output
    
Renderer = Blender