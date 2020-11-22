import bpy
import argparse
import sys
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
from glob import glob
import pandas as pd
from os import path
import os

from IPython import embed

IN2CM = 2.54

FOLDER = '/mnt/ssd/ycb-tools/models/ycb'
OUTPUT = '/mnt/ssd/ycb-tools/models/ycb_preprocessed'

FOLDER = '/data/'
OUTPUT = '/output/'


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

def rename(name):
    root = get_root()
    root.name = name
    root.data.name = name


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

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

def simplify_mesh():

    root = get_root()
    bpy.context.view_layer.objects.active = root
    dimensions = root.dimensions
    factor = max(dimensions) / 1000
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=factor)
    bpy.ops.mesh.decimate(ratio=0.5)
    bpy.ops.object.editmode_toggle()


def find_best_scale(current_dimensions, reference_dimensions):
    return (np.prod(reference_dimensions) / np.prod(current_dimensions))**(1/3.0)

def delete_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def get_root(type='MESH'):
    root = [x for x in bpy.context.scene.objects if x.type == type][0]
    return root

def join_object():
    bpy.context.view_layer.objects.active = get_root()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()

def apply_transformations():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def collapse_everything():
    apply_transformations()
    root = get_root()
    bpy.context.scene.collection.objects.unlink(root)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.context.scene.collection.objects.link(root)

def center_object():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    get_root().location = (0, 0, 0)


def rescale(reference_dimensions):
    print(reference_dimensions, get_root().dimensions)
    best_scale = find_best_scale(get_root().dimensions, 1)
    print(best_scale)
    get_root().scale = (best_scale, ) * 3

def load_file(fname):
    clear_memory()
    bpy.ops.wm.read_homefile()
    bpy.ops.file.autopack_toggle()
    delete_all()
    print(fname)
    if fname.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=fname)
    elif fname.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=fname)

def process_uid(uid, reference_dimensions):
    delete_all()
    try:

        output = path.join(args.output, f'{uid}.blend')

        if path.exists(output):
            return

        model_folder = path.join(args.folder, uid)

        model_size = get_size(model_folder) / 2**20

        # load_file(path.join(model_folder, 'scene.gltf'))
        # model_path = path.join(model_folder, 'google_16k', 'textured.obj')
        model_path = path.join(model_folder, 'google_64k', 'textured.obj')
        # model_path = path.join(model_folder, 'google_512k', 'textured.obj')
        if os.path.exists(model_path):
            load_file(model_path)
        else:
            return

        # join_object()
        center_object()
        # collapse_everything()

        # if model_size > 10:
            # simplify_mesh()

        # center_object()

        # rescale(reference_dimensions)
        # center_object()
        apply_transformations()
        rename(uid)
        bpy.ops.wm.save_as_mainfile(filepath=output)
    except:
        raise




if __name__ == '__main__':

    all_folders = glob(path.join(args.folder, '*'))

    for full_path in tqdm(all_folders[args.process_id::args.num_processes]):
        uid = path.basename(full_path)
        try:
            process_uid(uid, 100)
        except:
            raise

