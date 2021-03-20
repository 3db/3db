"""
Remove this
===========
"""

import importlib
import json
from collections import defaultdict
from os import path, remove
from glob import glob
from multiprocessing import cpu_count
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
import cv2
import numpy as np

NAME = 'Mitsuba'

try: # Do not fail if missing package for the master node
    import mitsuba
    import torch
    import enoki as ek
    import os
    import cv2

    mitsuba.set_variant('scalar_rgb')

    from mitsuba.core import Thread, Bitmap, Struct, ScalarTransform4f
    from mitsuba.core.xml import load_file, load_dict
    # from mitsuba.python.autodiff import render, write_bitmap, render_torch
    from mitsuba.python.util import traverse
except:
    pass


def load_attribute(content, folder):
    if isinstance(content, float):
        return content

    if isinstance(content, list):
        return {
            "type": "rgb",
            "value": content[:3]
        }
    elif isinstance(content, dict):
        if content['kind'] == 'texture':
            return {
                'type': 'bitmap',
                'filename': path.join(folder, content['path'])
            }
        if content['kind'] == 'vertex_attribute':
            return {
                'type': 'rgb',
                "value": [1, 1, 1]
            }

def average_roughness(roughness):
    if isinstance(roughness, float):
        return roughness

    if roughness['type'] == 'bitmap':
        img = cv2.imread(roughness['filename'])
        v = (img.mean() / 255.0) ** 2
        return v


def load_material(content, folder):
    plastic_part = {
        'type': 'roughplastic',
        'nonlinear': True,
        'distribution': 'ggx',
    }

    metal_part = {
        'type': 'roughconductor',
        'distribution': 'ggx',
    }

    reflectance = load_attribute(content['Base Color'], folder)
    roughness = load_attribute(content['Roughness'], folder)
    metalness = load_attribute(content['Metallic'], folder)

    plastic_part["diffuse_reflectance"] = reflectance
    plastic_part["alpha"] = average_roughness(roughness)
    metal_part["alpha"] = roughness
    metal_part["specular_reflectance"] = reflectance

    result = {
        'type': 'blendbsdf',
        'weight': metalness,
        '0_plastic': plastic_part,
        '1_metal': metal_part
    }

    if (isinstance(content['Normal'], dict)
            and content['Normal']['kind'] == 'texture'):

        result = {
            'type': 'normalmap',
            'normalmap': load_attribute(content['Normal'], folder),
            'bsdf': result
        }

    if content['Alpha'] != 1.0:
        result = {
            'type': 'mask',
            'opacity': content['Alpha'],
            'bsdf': result
        }

    return result


def load_madry_model(folder):
    result = {}
    for filename in glob(path.join(folder, '*.json')):
        with open(filename, 'r') as handle:
            object_id = path.basename(filename).replace('.json', '')
            content = json.load(handle)
            mesh_path = content['Mesh']
            full_mesh_path = path.join(folder, mesh_path)
            result["object_" + object_id] = {
                'type': 'ply',
                'filename': full_mesh_path,
                'bsdf': load_material(content, folder)
            }
    return result

def get_model_path(root_folder, model):
    return path.join(root_folder, 'mitsuba_models', model)

def get_env_path(root_folder, env):
    return path.join(root_folder, 'mitsuba_environments', env)

def enumerate_models(root_folder):
    return [path.basename(x) for x in glob(get_model_path(root_folder,
                                                          '*'))]

def enumerate_environments(root_folder):
    return [path.basename(x) for x in glob(get_env_path(root_folder,
                                                        '*.exr'))]

def load_env(root_folder, env):

    return {
        "type": "envmap",
        "scale": 1,
        "to_world": ScalarTransform4f.rotate(axis=[1, 0, 0],
                                             angle=90),
        "filename": get_env_path(root_folder, env)
    }

def load_model(root_folder, model):
    return model, load_madry_model(get_model_path(root_folder, model))


def get_model_uid(loaded_model):
    return loaded_model[0]

def setup_render(renderer_settings):
    if renderer_settings.cpu_cores is not None:
        mitsuba.core.set_thread_count(renderer_settings.cpu_cores)
    scene = {
        "type": "scene",
        "integrator": {
            'type': 'path'
        },
        "sensor": {
            "type": "perspective",
            "near_clip": 0.1,
            "far_clip": 1000.0,
            "to_world": ScalarTransform4f.look_at(origin=[1, 1, 0],
                                                  target=[0, 0, 0],
                                                  up=[0, 0, 1]),
            "film": {
                "type": "hdrfilm",
                "rfilter": {
                    "type": "gaussian"
                },
                "width": renderer_settings.resolution,
                "height": renderer_settings.resolution,
            },
            "sampler": {
                "type": "multijitter",
                "sample_count": renderer_settings.samples,
            },
        }
    }
    return scene


def render(uid, job, cli_args, renderer_settings, applier, loaded_model,
           loaded_env):

    loaded_model = loaded_model[1]

    renderer_settings = SimpleNamespace(**vars(cli_args),
                                        **renderer_settings)
    scene = setup_render(renderer_settings)

    scene['env_map'] = loaded_env
    scene.update(loaded_model)

    context = {
        'scene': scene,
        'object': loaded_model,
        'env': loaded_env
    }

    applier.apply_pre_controls(context)

    print(scene)

    loaded_scene = load_dict(scene)

    sensor = loaded_scene.sensors()[0]
    loaded_scene.integrator().render(loaded_scene, sensor)

    film = sensor.film()

    # Write out data as high dynamic range OpenEXR file

    img = film.bitmap(raw=True).convert(Bitmap.PixelFormat.RGBA, Struct.Type.UInt8, srgb_gamma=True)
    return np.array(img)
