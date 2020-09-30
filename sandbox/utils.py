import importlib
import cv2
import requests
import io
from urllib.parse import urljoin
from copy import deepcopy

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
                f"Attribute {k} unknown in {type(control).__name}")


def init_control(description):
    args = {}
    if 'args' in description:
        args = description['args']
    module = importlib.import_module(description['module'])
    control = module.Control(**args)
    d = {k: v for (k, v) in description.items() if k not in ['args', 'module']}
    overwrite_control(control,  d)
    return control


def init_policy(description):
    module = importlib.import_module(description['module'])
    return module.Policy(**{k: v for (k, v) in description.items() if k != 'module'})


def obtain_prediction(url, img, repeats=10):
    full_url = urljoin(url, "/predictions/sandbox-model")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".png", img)
    io_buf = io.BytesIO(buffer)

    for _ in range(repeats):
        try:
            result = requests.post(full_url, data=io_buf)
            if result.status_code == 200:
                predictions = result.json()
                ordered_prediction = sorted([(v, k) for (k, v) in predictions.items()])
                final_prediction = int(ordered_prediction[-1][1])
                return final_prediction
        except:
            raise
    return -1
