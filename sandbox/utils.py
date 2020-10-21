import importlib
import cv2
import requests
import io
from urllib.parse import urljoin
from copy import deepcopy
import torch as ch
from torchvision import transforms

def overwrite_control(control, data):

    # Make sure we are not overriding the dict containing the default values
    control.continuous_dims = deepcopy(control.continuous_dims)
    control.discrete_dims = deepcopy(control.discrete_dims)

    for k, v in data.items():
        if k in control.continuous_dims:
            control.continuous_dims[k] = v
        elif k in control.discrete_dims:
            control.discrete_dims[k] = v
        else:
            raise AttributeError(
                f"Attribute {k} unknown in {type(control).__name__}")


def init_control(description, root_folder):
    args = {}
    if 'args' in description:
        args = description['args']
    module = importlib.import_module(description['module'])
    control = module.Control(**args, root_folder=root_folder)
    d = {k: v for (k, v) in description.items() if k not in ['args', 'module']}
    overwrite_control(control,  d)
    return control


def init_policy(description):
    module = importlib.import_module(description['module'])
    return module.Policy(**{k: v for (k, v) in description.items() if k != 'module'})


def load_inference_model(args):
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        previous_context = ssl._create_default_https_context
        ssl._create_default_https_context = _create_unverified_https_context


    loaded_module = importlib.import_module(args['module'])
    model_args = args['args']

    model = getattr(loaded_module, args['class'])(**model_args)

    ssl._create_default_https_context = previous_context

    my_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args['resolution']),
        transforms.ToTensor(),
        transforms.Normalize(mean=args['normalization']['mean'],
                             std=args['normalization']['std'])
    ])

    def inference_function(image):
        image = ch.from_numpy(image)
        image = my_preprocess(image)
        image = image.unsqueeze(0)
        return model(image).data.numpy()[0]

    return inference_function

