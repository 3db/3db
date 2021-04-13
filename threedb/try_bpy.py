from types import SimpleNamespace
try:
    import bpy
except:
    # For type annotations
    bpy = SimpleNamespace(types=SimpleNamespace(
                            Scene=None,
                            Object=None,
                            Mesh=None))