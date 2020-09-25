import argparse
import colorsys
import pickle
import time
import json

from glob import glob
import pandas as pd
import sys
from copy import deepcopy
from collections import defaultdict
from os import path
import cv2
import bpy
import numpy as np
from uuid import uuid4
import bpycv
import mathutils
from multiprocessing import cpu_count
import os
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset')

    parser.add_argument('--focal-length', help='The focal length of the camera',
                        default=50, type=float)
    parser.add_argument('--aperture', help='The aperture in f-stops of the camera',
                        default=8, type=float)
    parser.add_argument('--gpu-id', help='The GPU to use to render (-1 for cpu)', 
                        default=-1, type=int,)
    parser.add_argument('--cpu-cores', help='number of cpu cores to use (default uses all)', 
                        default=None, type=int,)
    parser.add_argument('--resolution', help='The resolution of generated images',
                        default=256, type=int)
    parser.add_argument('--tile-size', help='The size of tiles used for GPU rendering',
                        default=256, type=int)
    parser.add_argument('--samples', help='The number of samples used when ray tracing',
                        default=512, type=int)
    parser.add_argument('--count', '-c', help="""Number of images to generate""",
                        type=int, default=100)
    parser.add_argument('--repeats', '-r', help="""Number of shots of the same config to take""",
                        type=int, default=5)
    parser.add_argument('--output', help='Folder where to write the generated data set',
                        type=str, required=True)
    parser.add_argument('--env', help='Which env to run',
                        type=str, required=True)
    parser.add_argument('--models', help='Location to the gltf models',
                        type=str, required=True)
    parser.add_argument('--metadata', help='Models metadata file',
                        type=str, required=True)
    parser.add_argument('--log', help='Where to store information about the shot',
                        type=str, required=True)
    parser.add_argument('--min-z', help='Minimum zoomout factor',
                        type=float, default=0.65)
    parser.add_argument('--max-z', help='Maximum zoomout factor',
                        type=float, default=0.85)
    parser.add_argument('--verbose', type=int, default=1, help='1 for verbose rendering')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume generation based on log.log file')

    try:
        index_of_sep = sys.argv.index('--')
        arguments = sys.argv[index_of_sep + 1:]
    except ValueError:
        arguments = []

    return parser.parse_args(arguments)


def load_object_info(args, available_models):
    object_to_class = {}
    class_to_object = defaultdict(list)

    for pkl_file in glob(path.join(args.metadata, '*')):
        clazz = pkl_file.replace('.pkl', '').split('class-')[1]

        try:
            with open(pkl_file, 'rb') as handle:
                result = pickle.load(handle)
        except:
            continue

        for uid, selected in result.items():
            if selected and uid in available_models:
                object_to_class[uid] = clazz
                class_to_object[clazz].append(uid)

    return list(class_to_object.keys()), object_to_class, class_to_object


def get_camera():
    return bpy.data.objects['Camera']


def lookat_viewport(target, location):
    diff = location - target
    diff = diff.normalized()
    rot_z = (np.arctan(diff.y / diff.x))
    if diff.x > 0:
        rot_z += np.pi / 2
    else:
        rot_z -= np.pi / 2
    return mathutils.Euler((np.arccos(diff[2]), 0, rot_z)).to_quaternion()


def sample_upper_sphere():
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    vec[2] = np.abs(vec[2])

    return mathutils.Vector(vec)


def focus_object(ob, camera, zoomout_factor):
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx = bpy.context.copy()
            space = area.spaces[0]
            r = space.region_3d
            ctx['area'] = area
            ctx['region'] = area.regions[-1]
            ctx['space_data'] = space
            ctx['camera'] = camera
            r.view_location = ob.location
            r.view_distance = 2
            previous_lens = space.lens
            space.lens = camera.data.lens * zoomout_factor
            direction = sample_upper_sphere()
            r.view_rotation = lookat_viewport(r.view_location,
                                              r.view_location + direction)
            bpy.ops.view3d.view_selected(ctx)
            bpy.ops.view3d.camera_to_view(ctx)
            space.lens = previous_lens
            camera.data.dof.focus_object = ob


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

def setup_camera(args):
    camera = get_camera()
    camera.data.lens = args.focal_length
    camera.data.dof.aperture_fstop = args.aperture

def import_object(args, uid):
    blendfile = path.join(args.models, uid + '.blend')
    section = "\\Object\\"
    object = uid

    filepath = uid + '.blend'
    directory = blendfile + section
    filename = object

    print(filepath)
    print(filename)
    print(directory)

    bpy.ops.wm.append(
        filepath=filepath, 
        filename=filename,
        directory=directory)

def shoot_object(uid, args, info, log_file_handle):

    try:
        # import_object(args, uid)
        bpy.ops.import_scene.gltf(filepath=path.join(args.models, uid + '.gltf'))
    except:
        return False

    ob = bpy.context.scene.objects[uid]

    bpy.context.scene.frame_set(1)

    for j in range(args.repeats):

        current_info = deepcopy(info)

        print(f"[Repeat {j}")
        print("[Focusing on object]")
        zoom_factor = np.random.uniform(args.min_z, args.max_z)
        focus_object(ob, get_camera(), zoom_factor)
        current_info['camera_position'] = np.array(get_camera().matrix_world).tolist()
        current_info['object_position'] = np.array(ob.matrix_world).tolist()
        current_info['zoom_factor'] = zoom_factor

        image_id = str(uuid4())

        current_info['image_id'] = image_id

        print("[Setting up cycles renderer]")
        setup_render(args)
        bpy.context.scene.render.filepath = path.join(args.output, image_id + '.jpeg')
        bpy.ops.render.render(use_viewport=False, write_still=True)
        print("[Render Done]")
        log_file_handle.write((json.dumps(current_info)) + "\n")

    bpy.ops.object.delete({"selected_objects": [ob]})
    clear_memory()
    return True

if __name__ == '__main__':

    args = parse_arguments()
    available_models = set([path.basename(x).replace('.gltf', '') for x in glob(path.join(args.models, '*'))])

    classes, object_to_class, class_to_object = load_object_info(args, available_models)
    with open('uid_to_class.json', 'w') as f:
        json.dump(object_to_class, f, indent=4)

    bpy.ops.wm.open_mainfile(filepath=args.env)

    if not args.verbose:
        # Disable all prints for running on the cluster
        os.close(1)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    setup_camera(args)

    taken = 0

    if args.resume and os.path.isfile(args.log):
        with open(args.log, 'r') as log_file_handle:
            taken = len(log_file_handle.readlines())
            if taken > 0: 
                print(f'[Resumung generation. {taken} samples already exist in {args.log}.]')
                taken //=args.repeats ## account for number of repeats

    with open(args.log, 'a+', buffering=1) as log_file_handle:
        while taken < args.count:
            selected_class = np.random.choice(classes)
            objects = class_to_object[selected_class]
            if len(objects) > 0:
                uid = np.random.choice(objects)
                print(uid, selected_class)
                info = {
                    'class': selected_class,
                    'args': vars(args),
                    'uuid': uid
                }
                if shoot_object(uid, args, info, log_file_handle):
                    taken += 1

    ## For testing if all the models
    # with open(args.log, 'a+', buffering=1) as log_file_handle:
    #     for uid in list(available_models):
    #         print(uid, object_to_class[uid])
    #         info = {
    #             'class': object_to_class[uid],
    #             'args': vars(args),
    #             'uuid': uid
    #         }
    #         if not shoot_object(uid, args, info, log_file_handle):
    #             print(f'{uid} FAILED')
    #             src_path = path.join(args.models, uid + '.gltf')
    #             shutil.move(src_path, path.abspath(os.path.join(src_path, os.pardir, os.pardir)))      
