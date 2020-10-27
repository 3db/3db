import numpy as np
from colorsys import hsv_to_rgb
from .base_control import BaseControl

class BackgroundControl(BaseControl):
    kind = 'post'

    continuous_dims = {
        'H': (0, 1),
        'S': (0, 1),
        'V': (0, 1),
    }

    discrete_dims = {}

    def apply(self, img, H, S, V):

        alpha = img[:, :, 3:].astype(float) / 255
        img = img[:, :, :3] * alpha + 255 * (1 - alpha) * np.array(hsv_to_rgb(H, S, V))[None, None]
        return np.uint8(img)
        

ControlBlender = BackgroundControl
