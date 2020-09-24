import bpy
import importlib
from collections import defaultdict
from os import path

def load_env(env):
    pass

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


def render(uid, job):

    control_list = job.control_order
    render_args = job.render_args
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
