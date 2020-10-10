import numpy as np
from PIL import Image

class BackgroundControl:
    kind = 'post'

    continuous_dims = {
        'R': (0, 1),
        'G': (0, 1),
        'B': (0, 1),
    }

    discrete_dims = {}

    def apply(self, img, R, G, B):

        fill_colour = (int(R * 255),
                        int(G * 255),
                        int(B * 255))

        bg = Image.new('RGBA', img.size, fill_colour)
        return Image.alpha_composite(bg, img)


Control = BackgroundControl
