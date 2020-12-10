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

    def __init__(self, output_channels, resolution=(256, 256), num_logits=1000, size=2500):
        self.image_buffers = {}

        for channel_name, channels, dtype in output_channels:
            buff = ch.zeros((size, channels, *resolution), dtype=dtype).share_memory_()
            self.image_buffers[channel_name] = buff

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
        image_results = {k: v[ix] for (k, v) in self.image_buffers.items()}
        return image_results, self.logits_buffer[ix], self.correct_buffer[ix].item()

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

    def allocate(self, images, logits, is_correct):
        ix = self.next_find_index()

        for channel_name, image_data in images.items():
            assert channel_name in self.image_buffers, "Unexpected channel " + channel_name
            self.image_buffers[channel_name][ix] = image_data

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
        return model(image).data.numpy()[0]

    return inference_function

