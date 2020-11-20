import numpy as np
from sandbox.controls.base_control import BaseControl

class DenoiseControl(BaseControl):
    kind = 'pre'

    continuous_dims = {}

    discrete_dims = {}

    def apply(self, context):
        import bpy
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'

BlenderControl = DenoiseControl
