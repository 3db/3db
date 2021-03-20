"""
threedb.controls.blender.denoiser
=================================

Defines the Blender Denoiser Control
"""

from typing import Any, Dict, List, Tuple
import bpy
from threedb.controls.base_control import PreProcessControl


class DenoiseControl(PreProcessControl):
    """Enable the built-in Denoise feature in blender

    Note
    ----
    This control has no actual parameter but is a way to enable this
    blender feature
    """

    def apply(self, context: Dict[str, Any], control_args: Dict[str, Any]) -> None:
        """Enable OPENIMAGEDENOISE denoiser

        Parameters
        ----------
        context : Dict[str, Any]
            The scene context
        """
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'


Control = DenoiseControl
