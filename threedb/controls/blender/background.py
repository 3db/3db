"""Defines the transparent background contro """

from colorsys import hsv_to_rgb
import torch as ch
from threedb.controls.base_control import BaseControl


class BackgroundControl(BaseControl):
    """Control that replace the transparent background of a render with a color

    Note
    ----

    This control needs transparent background. Therefore one need to have:

    `bpy.context.scene.render.film_transparent = True`

    However since this is a 'post' control it cannot set this parameter for
    the first render. The user has to make sure that a 'pre' control will
    set this parameter before the first render or the first image will be
    incorrect.
    """

    kind = 'post'

    continuous_dims = {
        'H': (0, 1),
        'S': (0, 1),
        'V': (0, 1),
    }

    discrete_dims = {}

    def apply(self, img: ch.Tensor, H: float, S: float, V: float) -> ch.Tensor:
        """Fill the alpha channel of an image with a HSV color

        Parameters
        ----------
        img
            The image to modify
        H
            Hue
        S
            Saturation
        V
            Value

        Returns
        -------
        The image with the background filled with the proper color
        """

        import bpy

        bpy.context.scene.render.film_transparent = True

        alpha = img[3:, :, :]
        img = img[:3, :, :] * alpha + (1 - alpha)
        img *= ch.tensor(hsv_to_rgb(H, S, V))[:, None, None].float()
        return img


BlenderControl = BackgroundControl
