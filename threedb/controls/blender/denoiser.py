"""
threedb.controls.blender.denoiser
=================================

Defines the Blender Denoiser Control
"""

from typing import Any, Dict, List, Tuple
from ...try_bpy import bpy
from threedb.controls.base_control import PreProcessControl


class DenoiseControl(PreProcessControl):
    """Enable the built-in Denoise feature in blender

    Note
    ----
    This control has no actual parameter but is a way to enable this
    blender feature. This control denoises renderings.

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/denoiser/images/image_1.png
            :width: 100
            :group: denoiser

        .. thumbnail:: /_static/logs/denoiser/images/image_2.png
            :width: 100
            :group: denoiser

        .. thumbnail:: /_static/logs/denoiser/images/image_3.png
            :width: 100
            :group: denoiser

        .. thumbnail:: /_static/logs/denoiser/images/image_4.png
            :width: 100
            :group: denoiser

        .. thumbnail:: /_static/logs/denoiser/images/image_5.png
            :width: 100
            :group: denoiser

        With denoising.

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/no_denoiser/images/image_1.png
            :width: 100
            :group: no_denoiser

        .. thumbnail:: /_static/logs/no_denoiser/images/image_2.png
            :width: 100
            :group: no_denoiser

        .. thumbnail:: /_static/logs/no_denoiser/images/image_3.png
            :width: 100
            :group: no_denoiser

        .. thumbnail:: /_static/logs/no_denoiser/images/image_4.png
            :width: 100
            :group: no_denoiser

        .. thumbnail:: /_static/logs/no_denoiser/images/image_5.png
            :width: 100
            :group: no_denoiser

        Without denoising.
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
    
    def unapply(self, context: Dict[str, Any]) -> None:
        bpy.context.scene.cycles.use_denoising = False

Control = DenoiseControl
