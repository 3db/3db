from types import SimpleNamespace
try:
    import bpy, mathutils
except:
    # For type annotations
    bpy = SimpleNamespace(types=SimpleNamespace(
                            Scene=None,
                            Object=None,
                            Mesh=None))

    mathutils = SimpleNamespace(Vector=SimpleNamespace(
                            Euler=None))                            