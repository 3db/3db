"""
threedb.controls.blender.imagenet_c
====================================

Defines the ImagenetCControl
"""

from typing import Any, Dict
import torch as ch
from imagenet_c import corrupt
from ..base_control import PostProcessControl

class ImagenetCControl(PostProcessControl):
    """
    Applies the ImageNet-C corruptions of [TODO].

    Discrete Dimensions:

    - ``corruption_name``: The name of corruption that will be applied (see
      `here <TODO>`_ for list of corruption names)
    - ``severity``: Imagenet-C severity parameter. (range: {0, 1, 2, 3, 4, 5})
    """
    def __init__(self, root_folder: str):
        discrete_dims = {
            'severity': [1, 2, 3, 4, 5],
            'corruption_name': ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                'defocus_blur', 'glass_blur', 'motion_blur',
                                'zoom_blur', 'snow', 'frost', 'fog',
                                'speckle_noise', 'gaussian_blur', 'spatter',
                                'saturate', 'brightness', 'contrast',
                                'elastic_transform', 'pixelate',
                                'jpeg_compression']
        }
        super().__init__(root_folder,
                         discrete_dims=discrete_dims)

    def apply(self, render: ch.Tensor, control_args: Dict[str, Any]) -> ch.Tensor:
        """Apply an Imagenet-C corruption on the rendered image.

        Parameters
        ----------
        render : ch.Tensor
            Image to transform.
        control_args : Dict[str, Any]
            Corruption parameterization, must have keys ``corruption_name`` and
            ``severity`` (see class documentation for information about the
            control arguments).

        Returns
        -------
        ch.Tensor
            The transformed image.
        """
        no_err, msg = self.check_arguments(control_args)
        assert no_err, msg

        sev, c_name = control_args['severity'], control_args['corruption_name']
        img = render.numpy()
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype('uint8')
        img = corrupt(img, severity=sev, corruption_name=c_name)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32') / 255
        return ch.from_numpy(img)

Control = ImagenetCControl
