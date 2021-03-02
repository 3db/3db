import importlib
import ssl
from copy import deepcopy
from multiprocessing import Queue
from queue import Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch as ch
from torch.types import _dtype
from torchvision import transforms


class BigChungusCyclicBuffer:
    """
    A concurrent cyclic buffer with reference counting to store the result
    and avoid copying them to every sub process.


    """
    def __init__(self, buffers: Optional[Dict[str,Tuple[List[int], _dtype]]],
                       size: int = 1001) -> None:
        # self.image_buffers = {}
        self.buffers = {}
        
        for buf_name, (buf_size, buf_dtype) in buffers.items():
            buf = ch.zeros((size, *buf_size), dtype=buf_dtype).share_memory_()
            self.buffers[buf_name] = buf
            # self.buffers['images'][buf_name] = buf

        # self.outputs_buffer = ch.zeros((size, *output_shape), dtype=ch.float32).share_memory_()
        # self.correct_buffer = ch.zeros(size, dtype=ch.uint8).share_memory_()
        
        self.used_buffer = np.zeros(size, dtype='uint8')
        self.free_idx = list(range(size))
        self.first = 0
        self.last = 0
        self.size = size
        self.mask = 0
        self.registration_count = 0
        self.events = Queue()

    def __getitem__(self, ind: int) -> Dict[str, ch.Tensor]:
        return {k: v[ind] for (k, v) in self.buffers.items()}
        # image_results = {k: v[ix] for (k, v) in self.image_buffers.items()}
        # return image_results, self.outputs_buffer[ix], self.correct_buffer[ix].item()

    def free(self, ind: int, reg_id: int):
        self.events.put((ind, reg_id))

    def register(self):
        self.registration_count += 1
        if self.registration_count > 8:
            raise Exception("Too many registrations")
        self.mask = 2**self.registration_count - 1
        return self.registration_count

    def process_events(self):
        while True:
            try:
                (event, reg_id) = self.events.get(block=False)
                assert self.used_buffer[event] > 0
                if reg_id == -1:
                    self.used_buffer[event] = 0
                else:
                    self.used_buffer[event] ^= 1 << (reg_id - 1)
                if self.used_buffer[event] == 0:
                    self.free_idx.append(event)
            except Empty:
                break

    def next_find_index(self) -> int:
        while True:
            self.process_events()
            try:
                ind = self.free_idx.pop()
                assert self.used_buffer[ind] == 0
                self.used_buffer[ind] = self.mask
                return ind
            except IndexError:
                # Should we add some kind of logging here?
                np.save('/tmp/used_buffer.npy', self.used_buffer)

    # def allocate(self, images, outputs, is_correct):
    def allocate(self, data: Dict[str, ch.Tensor]):
        next_ind = self.next_find_index()

        for buf_key, buf_data in data.items():
            assert buf_key in self.buffers, "Unexpected channel " + buf_key
            assert buf_data.dtype == self.buffers[buf_key].dtype, \
                f"Expected datatype {self.buffers[buf_key].dtype}, got {buf_data.dtype}"
            self.buffers[buf_key][next_ind] = buf_data

        return next_ind

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

    control_module = getattr(module, f"{engine_name.capitalize()}Control")
    control = control_module(**args, root_folder=root_folder)
    filtered_desc = {k: v for (k, v) in description.items() if k not in ['args', 'module']}
    overwrite_control(control,  filtered_desc)
    return control


def init_policy(description):
    module = importlib.import_module(description['module'])
    return module.Policy(**{k: v for (k, v) in description.items() if k != 'module'})


def load_inference_model(args):
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

    def resize(tens):
        return ch.nn.functional.interpolate(tens[None], size=args['resolution'], mode='bilinear')[0]

    my_preprocess = transforms.Compose([
        resize,
        transforms.Normalize(mean=args['normalization']['mean'],
                             std=args['normalization']['std'])
    ])

    def inference_function(image):
        image = my_preprocess(image)
        return model(image.unsqueeze(0))[0], image.shape

    return inference_function
