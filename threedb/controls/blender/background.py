"""
threedb.controls.blender.background
===================================

Set the background to a solid-color. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/background.yaml>`_.
"""

from typing import Any, Dict
from colorsys import hsv_to_rgb
import torch as ch

from ...try_bpy import bpy
from ..base_control import PostProcessControl

class BackgroundControl(PostProcessControl):
    """Control that replace the transparent background of a render (i.e., the
    alpha channel) with a given color specified in HSV by the control parameters.

    Continuous parameters:

    - ``H``, ``S`` and ``V``: the hue, saturation, and value of the color to
      fill the background with. (range: ``[0, 1]``)

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/background/images/image_1.png
            :width: 100
            :group: background

        .. thumbnail:: /_static/logs/background/images/image_2.png
            :width: 100
            :group: background

        .. thumbnail:: /_static/logs/background/images/image_3.png
            :width: 100
            :group: background

        .. thumbnail:: /_static/logs/background/images/image_4.png
            :width: 100
            :group: background

        .. thumbnail:: /_static/logs/background/images/image_5.png
            :width: 100
            :group: background
        
        Varying all parameters across their ranges.
    """

    def __init__(self, root_folder: str):
        continuous_dims = {
            'H': (0., 1.),
            'S': (0., 1.),
            'V': (0., 1.),
        }
        super().__init__(root_folder,
                         continuous_dims=continuous_dims)

    def apply(self, render: ch.Tensor, control_args: Dict[str, Any]) -> ch.Tensor:
        check_result = self.check_arguments(control_args)
        assert check_result[0], check_result[1]
        bpy.context.scene.render.film_transparent = True

        rgb_color = hsv_to_rgb(control_args['H'], control_args['S'], control_args['V'])
        rgb_color = ch.tensor(rgb_color)[:, None, None].float()
        alpha = render[3:, :, :]
        img = render[:3, :, :] * alpha + (1 - alpha) * rgb_color
        return img

Control = BackgroundControl
