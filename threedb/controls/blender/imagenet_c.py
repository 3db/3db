"""
threedb.controls.blender.imagenet_c
====================================

Defines the ImagenetCControl
"""

from typing import Any, Dict, List, Tuple
import torch as ch
from imagenet_c import corrupt
from ..base_control import PostProcessControl

class ImagenetCControl(PostProcessControl):
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
    @property
    def continuous_dims(self) -> Dict[str, Tuple[float, float]]:
        return {
            'severity': (0, 1),
        }

    @property
    def discrete_dims(self) -> Dict[str, List[Any]]:
        return {
            'corruption_name': ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                'defocus_blur', 'glass_blur', 'motion_blur',
                                'zoom_blur', 'snow', 'frost', 'fog',
                                'speckle_noise', 'gaussian_blur', 'spatter',
                                'saturate', 'brightness', 'contrast',
                                'elastic_transform', 'pixelate',
                                'jpeg_compression']
        }

    def apply(self, render: ch.Tensor, control_args: Dict[str, Any]) -> ch.Tensor:
        """Apply an Imagenet-C corruption on the rendered image.

        Parameters
        ----------
        render : ch.Tensor
            Image to transform.
        control_args : Dict[str, Any]
            Corruption parameterization, must have keys ``corruption_name`` and
            ``severity``.

        Returns
        -------
        ch.Tensor
            The transformed image.
        """
        args_check = self.check_arguments(control_args)
        assert args_check[0], args_check[1]

        sev, c_name = control_args['severity'], control_args['corruption_name']
        img = render.numpy()
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype('uint8')
        img = corrupt(img, severity=sev, corruption_name=c_name)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32') / 255
        return ch.from_numpy(img)

BlenderControl = ImagenetCControl
