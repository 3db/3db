"""Defines the ImagenetCControl"""

import torch as ch
from threedb.controls.base_control import BaseControl
from imagenet_c import corrupt


class ImagenetCControl(BaseControl):
    """

    Continuous Dimensions
    ---------------------
    severity
        Imagenet-C severity parameter

    Discrete Dimensions
    -------------------
    corruption_name
        The name of corruption that will be applied

    Note
    ----
    To know the list of all the possible values, take a look at the default
    value of the corruption_name.
    """
    kind = 'post'

    continuous_dims = {
        'severity': (0, 1),
    }

    discrete_dims = {
        'corruption_name': ['gaussian_noise', 'shot_noise', 'impulse_noise',
                            'defocus_blur', 'glass_blur', 'motion_blur',
                            'zoom_blur', 'snow', 'frost', 'fog',
                            'speckle_noise', 'gaussian_blur', 'spatter',
                            'saturate', 'brightness', 'contrast',
                            'elastic_transform', 'pixelate',
                            'jpeg_compression']
    }

    def apply(self, img: ch.Tensor, severity: float, corruption_name: str):
        """Apply an Imagenet-C corruption on the rendered image

        Parameters
        ----------
        img
            The image to be modified
        severity
            Imagenet-C severity parameter
        corruption_name
            The name of corruption that will be applied
        """

        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype('uint8')
        img = corrupt(img, severity=severity, corruption_name=corruption_name)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32') / 255
        return ch.from_numpy(img)


BlenderControl = ImagenetCControl
