import argparse
from glob import glob
from os import path

import unreal


parser = argparse.ArgumentParser()
parser.add_argument('folder', help='The folder that contains all the models')
args = parser.parse_args()

print(args.folder)
model_folders = glob(path.join(args.folder, '*'))
print(len(model_folders))

factory = unreal.GLTFImportFactory()

taskList = unreal.Array(unreal.AssetImportTask)

for i, model_folder in list(enumerate(model_folders))[0:100]:
    matching_gltf = glob(path.join(model_folder, '*.gltf'))
    if len(matching_gltf) != 1:
        print(model_folders)
        continue
    else:
        matching_gltf = matching_gltf[0]

    task = unreal.AssetImportTask()
    task.set_editor_property('filename', matching_gltf)
    task.set_editor_property('automated', True)
    task.set_editor_property('options', unreal.DatasmithGLTFImportOptions())
    task.options.set_editor_property('import_scale', 1)
    task.options.set_editor_property('generate_lightmap_u_vs', True)
    task.set_editor_property('destination_name', "Super test")
    task.set_editor_property('destination_path', "/Game/models")
    task.set_editor_property('replace_existing_settings', False)
    task.set_editor_property('replace_existing', False)
    taskList.append(task)


asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
asset_tools.import_asset_tasks(taskList)
unreal.SystemLibrary.collect_garbage()
