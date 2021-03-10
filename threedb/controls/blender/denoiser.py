"""Defines the Blender Denoiser Control"""

from threedb.controls.base_control import BaseControl


class DenoiseControl(BaseControl):
    """Enable the built-in Denoise feature in blender

    Note
    ----
    This control has no actual parameter but is a way to enable this
    blender feature
    """
    kind = 'pre'

    continuous_dims = {}

    discrete_dims = {}

    def apply(self, context):
        """Enable OPENIMAGEDENOISE denoiser

        Parameters
        ----------
        context
            The scene context
        """
        import bpy
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'


BlenderControl = DenoiseControl
