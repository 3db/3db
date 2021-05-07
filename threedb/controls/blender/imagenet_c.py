"""
threedb.controls.blender.imagenet_c
====================================

Apply corruptions to renderings. An example config file using this control can be found here:
`<https://github.com/3db/3db/tree/main/examples/unit_tests/imagenet_c.yaml>`_.
"""

from typing import Any, Dict
import torch as ch
from imagenet_c import corrupt
from ..base_control import PostProcessControl

class ImagenetCControl(PostProcessControl):
    """
    Applies the ImageNet-C corruptions from `<https://github.com/hendrycks/robustness>`_.

    Discrete Dimensions:

    - ``corruption_name``: The name of corruption that will be applied. Includes
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'speckle_noise', 'gaussian_blur', 'spatter',
        'saturate', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'.
    - ``severity``: Imagenet-C severity parameter. (range: ``{0, 1, 2, 3, 4, 5}``)

    .. admonition:: Example images

        .. thumbnail:: /_static/logs/imagenet_c/images/image_1.png
            :width: 100
            :group: imagenet_c

        .. thumbnail:: /_static/logs/imagenet_c/images/image_2.png
            :width: 100
            :group: imagenet_c

        .. thumbnail:: /_static/logs/imagenet_c/images/image_3.png
            :width: 100
            :group: imagenet_c

        .. thumbnail:: /_static/logs/imagenet_c/images/image_4.png
            :width: 100
            :group: imagenet_c

        .. thumbnail:: /_static/logs/imagenet_c/images/image_5.png
            :width: 100
            :group: imagenet_c
        
        Examples of impulse noise application at various severity levels.
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


    def unapply(self, context: Dict[str, Any]) -> None:
        pass

Control = ImagenetCControl
