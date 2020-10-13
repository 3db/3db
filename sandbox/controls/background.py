import numpy as np

class BackgroundControl:
    kind = 'post'

    continuous_dims = {
        'R': (0, 1),
        'G': (0, 1),
        'B': (0, 1),
    }

    discrete_dims = {}

    def apply(self, img, R, G, B):

        alpha = img[:, :, 3:].astype(float) / 255
        img = img[:, :, :3] * alpha + 255 * (1 - alpha) * np.array([R, G, B])[None, None]
        return np.uint8(img)
        

Control = BackgroundControl
