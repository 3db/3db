import bpy
import math
import random

light_object = bpy.context.scene.objects["Light"]
light = bpy.context.scene.objects["Light"].data
obj = bpy.data.objects["Cube"]

# Let's sample a point on a sphere of desired radius
radius = 10
import numpy as np

a = np.random.randn(3)

# Optional - Constrain lamp to be only on upper hemisphere
a[2] = 1.0
a = a / np.linalg.norm(a)

# Set light location to point on sphere around object
light_object.location = radius * a

# Change light properties
light.type = "POINT"
light.color = (1.0, 0.10, 0.6)  # RGB color of light
light.energy = 1000.0

# Optional - point light towards object
bpy.ops.object.constraint_add(type="TRACK_TO")
bpy.context.object.constraints["Track To"].target = obj
