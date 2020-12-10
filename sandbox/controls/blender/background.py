import torch as ch
from colorsys import hsv_to_rgb
from sandbox.controls.base_control import BaseControl

class BackgroundControl(BaseControl):
    kind = 'post'

    continuous_dims = {
        'H': (0, 1),
        'S': (0, 1),
        'V': (0, 1),
    }

    discrete_dims = {}

    def apply(self, img, H, S, V):
        import bpy

        bpy.context.scene.render.film_transparent = True

        alpha = img[3:, :, :]
        img = img[:3, :, :] * alpha + (1 - alpha) * ch.tensor(hsv_to_rgb(H, S, V))[:, None, None].float()
        return img


BlenderControl = BackgroundControl
