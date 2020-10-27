import os

image_file = os.getenv("texture")
if not image_file:
    image_file = "test_texture.png"

PIC_obj = "Cube"

import bpy
from bpy_extras.image_utils import load_image

change_texture = False
transparent = True

mat_properties = {"metallic": 0, "specular": 0, "roughness": 0.5, "alpha": 0.7}

mat = bpy.data.materials.new(name="test")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]

if change_texture:
    # Create texture object and load image
    texImage = mat.node_tree.nodes.new("ShaderNodeTexImage")
    texImage.image = bpy.data.images.load(image_file)

    # Connect color of image to Base color of material
    mat.node_tree.links.new(bsdf.inputs["Base Color"], texImage.outputs["Color"])

bsdf.inputs["Metallic"].default_value = mat_properties["metallic"]
bsdf.inputs["Specular"].default_value = mat_properties["specular"]
bsdf.inputs["Roughness"].default_value = mat_properties["roughness"]
bsdf.inputs["Transmission"].default_value = mat_properties["alpha"]

ob = bpy.context.scene.objects["Cube"]

# Assign it to object
if ob.data.materials:
    ob.data.materials[0] = mat
else:
    ob.data.materials.append(mat)

if transparent:
    ob.active_material.blend_method = "BLEND"

