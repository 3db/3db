"""
threedb.controls.blender.denoiser
=================================

Denoise renderings. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/denoiser.yaml>`_. (And
of the other controls in the ``unit_tests`` folder, this files parent folder, use
this control as well for more realistic renderings).

"""

from typing import Any, Dict, List, Tuple
from ...try_bpy import bpy
from threedb.controls.base_control import PreProcessControl


class DenoiseControl(PreProcessControl):
    """Enable the built-in Denoise feature in blender

    Note
    ----
    This control enables the blender feature for denoising renderings.
    It has no parameters.

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
