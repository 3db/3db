import bpy
import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
from glob import glob
import pandas as pd
from os import path
import os
from os import path, makedirs, remove
import shutil
from uuid import uuid4
import json

texture_exts = {
    'BMP': '.bmp',
    'HDR': '.hdr',
    'JPEG': '.jpg',
    'JPEG2000': '.jpg',
    'PNG': '.png',
    'OPEN_EXR': '.exr',
    'OPEN_EXR_MULTILAYER': '.exr',
    'TARGA': '.tga',
    'TARGA_RAW': '.tga',
}


FOLDER = '/mnt/ssd/sandbox_data/OSimModelsPreProcessed'
OUTPUT = '/mnt/ssd/sandbox_data/MitsubaModels/'


parser = argparse.ArgumentParser(description='Pre preocess gltf models')
parser.add_argument('--folder', default=FOLDER, help='Where the source models are')
parser.add_argument('--output', default=OUTPUT, help='Where to write the models')
parser.add_argument('--num-processes', default=1, type=int,
                    help='Total number of processes you intened to spawn')
parser.add_argument('--process-id', default=0, type=int,
                    help='The id of the current process [0 num-processes - 1]')

argv = sys.argv
argv = argv[argv.index("--") + 1:]
args = parser.parse_args(argv)

def apply_transformations(object):
    bpy.ops.object.select_all(action='DESELECT')
    object.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def removeMeshFromMemory(mesh):
    try:
        mesh.user_clear()
        can_continue = True
    except:
        can_continue = False

    if can_continue:
        try:
            bpy.data.meshes.remove(mesh)
            result = True
        except:
            result = False
    else:
        result = False

    return result


def clear_memory():
    for mesh in bpy.data.meshes:
        removeMeshFromMemory(mesh)


def delete_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def get_root(type='MESH'):
    root = [x for x in bpy.context.scene.objects if x.type == type][0]
    return root


def load_file(fname):
    clear_memory()
    bpy.ops.wm.read_homefile()
    delete_all()
    print(fname)
    bpy.ops.import_scene.gltf(filepath=fname)


def rm_r(p):
    if path.isdir(p) and not path.islink(p):
        shutil.rmtree(p)
    elif path.exists(p):
        remove(p)

def make_clear_folder(p):
    rm_r(p)
    makedirs(p)
    return p


class OutputContext:

    def __init__(self, root_folder, object_id):
        full_folder = make_clear_folder(path.join(root_folder, object_id))
        self.full_folder = full_folder + '/'
        self.textures_folder = make_clear_folder(path.join(full_folder, 'textures'))
        self.meshes_folder = make_clear_folder(path.join(full_folder, 'meshes'))

    def save_mesh(self, mesh):
        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(True)
        meshid = str(uuid4())
        mesh_path = path.join(self.meshes_folder, meshid + '.ply')
        bpy.ops.export_mesh.ply(filepath=mesh_path, use_ascii=False, use_selection=True)
        return mesh_path.replace(self.full_folder, '')


    def save_image(self, fname, content):
        print("###", self.textures_folder)
        full_path = path.join(self.textures_folder, fname)
        cv2.imwrite(full_path, content)
        return full_path.replace(self.full_folder, '')

    def save_info(self, info):
        fname = path.join(self.full_folder, str(uuid4()) + '.json')

        with open(fname, 'w+') as handle:
            json.dump(info, handle, indent=2)

        return fname.replace(self.full_folder, '')

def process_texture(node, context, channels='ALL'):
    try:
        image = node.image

        new_imagename = str(uuid4()) + texture_exts[image.file_format]
        image.filepath = path.join('/tmp', new_imagename)
        try:
            image.save() # making sure the image is saved to disc
        except:
            return {
                'kind': 'error'
            }
        img = cv2.imread(image.filepath, cv2.IMREAD_COLOR)
        img = cv2.flip(img, 0)
        remove(image.filepath)
        print(img.shape)
        # CV2 uses BGR and not RGB
        if channels == 'B':
            img = img[:, :, 0]
        elif channels == 'G':
            img = img[:, :, 1]
        elif channels == 'R':
            img = img[:, :, 2]

        return {
            'kind': 'texture',
            'path': context.save_image(new_imagename, img)
        }
    except:  # This is probably a vertex attribute
        if node.type == 'MATH':
            return {
                'kind': 'error'
            }
        if node.type == 'MIX_RGB':
            return process_texture(node.inputs[1].links[0].from_node, context, channels)
        return {
            'kind': 'vertex_attribute',
            'attr_name': node.layer_name
        }

def dump_link(link, context):
    node = link.from_node
    channels = 'ALL'

    if node.name == 'Separate RGB':
        channels = link.from_socket.name
        node = node.inputs[-1].links[0].from_node

    if node.name == 'Normal Map':
        node = node.inputs[-1].links[0].from_node

    return process_texture(node, context, channels)

def dump_input(socket, context):
    pname = socket.name
    is_roughtness = "Roughness" in pname
    if socket.is_linked:
        return dump_link(socket.links[0], context)
    else:
        try:
            value = float(socket.default_value)
            if is_roughtness:
                value = value ** 2
        except:
            value = list(socket.default_value)
        return value

def split_object(object):
    bpy.ops.object.select_all(action='DESELECT')
    object.select_set(True)
    bpy.ops.mesh.separate(type='MATERIAL')

def inspect_object(object, output_dir):
    ob_id = object.name
    context = OutputContext(output_dir, ob_id)
    mat_count = len(object.material_slots)

    max_dim = max([x.co.length for x in object.data.vertices])
    object.scale /= max_dim * 2
    apply_transformations(object)

    if mat_count > 1:
        split_object(object)

    for obj in bpy.data.objects:
        if obj.name.startswith(ob_id):
            mesh_data = context.save_mesh(obj)
            assert len(obj.material_slots) == 1
            material = obj.material_slots[0].material
            output_node = material.node_tree.get_output_node('CYCLES')
            bsdf =  output_node.inputs[0].links[0].from_node
            material_info = {
                'Mesh': mesh_data
            }
            for parameter in bsdf.inputs:
                r = dump_input(parameter, context)
                if r is not None:
                    material_info[parameter.name] = r

            context.save_info(material_info)
            print(material.name, mesh_data, material_info)


def process_uid(full_path, output_dir):

    delete_all()
    try:
        load_file(full_path)
        obj = get_root()
        inspect_object(obj, output_dir)

    except:
        raise




if __name__ == '__main__':

    all_files = glob(path.join(args.folder, '*.gltf'))

    for full_path in tqdm(all_files[args.process_id::args.num_processes]):
        try:
            process_uid(full_path, args.output)
        except:
            raise

