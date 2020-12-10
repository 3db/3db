import importlib
import numpy as np
import cv2
import requests
import io
from urllib.parse import urljoin
from copy import deepcopy
import torch as ch
from torchvision import transforms
import numpy as np
from multiprocessing import Queue
from queue import Empty


# Concurrent cyclic buffer with reference counting to store the result
# and avoid copying them to every sub process
class BigChungusCyclicBuffer:

    def __init__(self, resolution=(256, 256), num_logits=1000, size=5000):
        self.image_buffer = ch.zeros((size, 3, *resolution), dtype=ch.float32).share_memory_()
        self.logits_buffer = ch.zeros((size, num_logits), dtype=ch.float32).share_memory_()
        self.correct_buffer = ch.zeros(size, dtype=ch.uint8).share_memory_()
        self.used_buffer = np.zeros(size, dtype='uint8')
        self.free_idx = list(range(size))
        self.first = 0
        self.last = 0
        self.size = size
        self.registration_count = 0
        self.events = Queue()

    def __getitem__(self, ix):
        return self.image_buffer[ix], self.logits_buffer[ix], self.correct_buffer[ix].item()

    def free(self, ix):
        self.events.put(ix)

    def register(self):
        self.registration_count += 1

    def process_events(self):
        while True:
            try:
                event = self.events.get(block=False)
                assert self.used_buffer[event] > 0
                self.used_buffer[event] -= 1
                if self.used_buffer[event] == 0:
                    self.free_idx.append(event)
            except Empty:
                break

    def next_find_index(self):
        while True:
            self.process_events()
            try:
                ix = self.free_idx.pop()
                assert self.used_buffer[ix] == 0
                self.used_buffer[ix] = self.registration_count
                return ix
            except IndexError:
                pass

    def allocate(self, image, logits, is_correct):
        ix = self.next_find_index()
        self.image_buffer[ix] = image
        self.logits_buffer[ix] = logits
        self.correct_buffer[ix] = is_correct
        return ix


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


def init_control(description, root_folder, engine_name):
    args = {}
    if 'args' in description:
        args = description['args']
    full_module_path = description['module']

    try:
        module = importlib.import_module(full_module_path)
    except ModuleNotFoundError:
        full_module_path = f"sandbox.controls.{engine_name.lower()}.{full_module_path}"
        module = importlib.import_module(full_module_path)

    Control = getattr(module, f"{engine_name.capitalize()}Control")
    control = Control(**args, root_folder=root_folder)
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
    model.eval()

    ssl._create_default_https_context = previous_context

    def resize(x):
        x = x.unsqueeze(0)
        x = ch.nn.functional.interpolate(x, size=args['resolution'], mode='bilinear')
        return x[0]

    my_preprocess = transforms.Compose([
        resize,
        transforms.Normalize(mean=args['normalization']['mean'],
                             std=args['normalization']['std'])
    ])

    def inference_function(image):
        image = my_preprocess(image)
        image = image.unsqueeze(0)
        return model(image)[0]

        """
        if isinstance(out, list): 
            # Object detection
            N = out[0]['boxes'].shape[0]
            results = [out[0][s].float().view(N, -1) for s in ('boxes', 'labels', 'scores')]
            out = ch.cat(results, dim=1)
            return out
        elif isinstance(out, ch.tensor):
            # Image classification
            return out.data.numpy()[0]
        """

    return inference_function