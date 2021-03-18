"""Defines the transparent background control"""

from typing import Any, Dict, Tuple
from colorsys import hsv_to_rgb
import torch as ch

import bpy
from ..base_control import PostProcessControl

class BackgroundControl(PostProcessControl):
    """Control that replace the transparent background of a render with a color

    Note
    ----

    This control needs transparent background. Therefore one need to have:

    `bpy.context.scene.render.film_transparent = True`

    However since this is a 'post-processing control it cannot set this
    parameter for the first render. The user has to make sure that a 'pre'
    control will set this parameter before the first render or the first image
    will be incorrect.
    """

    @property
    def continuous_dims(self) -> Dict[str, Tuple[float, float]]:
        return {
            'H': (0, 1),
            'S': (0, 1),
            'V': (0, 1),
        }

    def apply(self, render: ch.Tensor, control_args: Dict[str, Any]) -> ch.Tensor:
        """Fill the alpha channel of an image with a HSV color.

        Parameters
        ----------
        render : ch.Tensor
            The image to modify
        control_args : Dict[str, Any]
            Must have keys ``H``, ``S``, and ``V`` which dictate the color that
            will be used to fill in the background.

        Returns
        -------
        ch.Tensor
            The image with the background filled with the proper color.
        """
        check_result = self.check_arguments(control_args)
        assert check_result[0], check_result[1]
        bpy.context.scene.render.film_transparent = True

        alpha = render[3:, :, :]
        img = render[:3, :, :] * alpha + (1 - alpha)
        rgb_color = hsv_to_rgb(control_args['H'], control_args['S'], control_args['V'])
        img *= ch.tensor(rgb_color)[:, None, None].float()
        return img