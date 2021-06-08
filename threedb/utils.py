"""
threedb.utils
=============
"""

import importlib
import ssl
from copy import deepcopy
from multiprocessing import Queue
from queue import Empty
from threedb.controls.base_control import BaseControl
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch as ch
from torch import Tensor
from torch.types import _dtype
from torchvision import transforms
from tqdm import tqdm

def str_to_dtype(dtype_str: str) -> _dtype:
    return getattr(ch, dtype_str)

class CyclicBuffer:
    """
    A concurrent cyclic buffer with reference counting to store the result
    and avoid copying them to every sub process.


    """
    def __init__(self, buffers: Optional[Dict[str, Tuple[List[int], str]]] = None,
                 size: int = 1001, with_tqdm: bool = True) -> None:
        self.buffers = {}
        if buffers is None:
            buffers = {}
            self.initialized = False
        else:
            self.declare_buffers(buffers)

        self.used_buffer = np.zeros(size, dtype='uint8')
        self._free_idx = list(range(size))
        self.size = size

        self.first = 0
        self.last = 0
        self.mask = 0
        self.registration_count = 0
        self.events = Queue()

        self.progress_bar = None
        if with_tqdm:
            self.progress_bar = tqdm(unit='slots', desc='Buffer left',
                                     total=self.size, smoothing=0)

    def declare_buffers(self, buffers: Dict[str, Tuple[List[int], str]]) -> bool:
        if self.initialized and (buffers != self.declared_buffers):
            return False
        elif self.initialized:
            return True
        
        self.initialized = True
        self.declared_buffers = buffers
        for buf_name, (buf_size, buf_dtype) in buffers.items():
            buf = ch.zeros((self.size, *buf_size), dtype=str_to_dtype(buf_dtype)).share_memory_()
            self.buffers[buf_name] = buf
        return True

    def __getitem__(self, ind: int) -> Dict[str, ch.Tensor]:
        assert self.initialized, 'Buffer has not been initialized'
        return {k: v[ind] for (k, v) in self.buffers.items()}

    def free(self, ind: int, reg_id: int):
        assert self.initialized, 'Buffer has not been initialized'
        self.events.put((ind, reg_id))

    def register(self):
        self.registration_count += 1
        if self.registration_count > 8:
            raise Exception("Too many registrations")
        self.mask = 2**self.registration_count - 1
        return self.registration_count

    def process_events(self):
        assert self.initialized, 'Buffer has not been initialized'
        while True:
            try:
                (event, reg_id) = self.events.get(block=False)
                assert self.used_buffer[event] > 0
                if reg_id == -1:
                    self.used_buffer[event] = 0
                else:
                    self.used_buffer[event] ^= 1 << (reg_id - 1)
                if self.used_buffer[event] == 0:
                    self._free_idx.append(event)
                    if self.progress_bar is not None:
                        self.progress_bar.update(-1)
            except Empty:
                break

    def next_find_index(self) -> int:
        assert self.initialized, 'Buffer has not been initialized'
        while True:
            self.process_events()
            try:
                ind = self._free_idx.pop()
                if self.progress_bar is not None:
                    self.progress_bar.update(1)
                assert self.used_buffer[ind] == 0
                self.used_buffer[ind] = self.mask
                return ind
            except IndexError:
                # Should we add some kind of logging here?
                np.save('/tmp/used_buffer.npy', self.used_buffer)

    def allocate(self, data: Dict[str, ch.Tensor]):
        assert self.initialized, 'Buffer has not been initialized'
        next_ind = self.next_find_index()

        for buf_key, buf_data in data.items():
            assert buf_key in self.buffers, "Unexpected channel " + buf_key
            assert buf_data.dtype == self.buffers[buf_key].dtype, \
                f"Expected datatype {self.buffers[buf_key].dtype}, got {buf_data.dtype} for key {buf_key}"
            self.buffers[buf_key][next_ind] = buf_data

        return next_ind

    def close(self):
        if self.progress_bar is not None:
            self.progress_bar.close()

def overwrite_control(control: BaseControl, data: Dict[str, Union[Tuple[float, float], List[Any]]]):
    for key, val in data.items():
        if key in control.continuous_dims:
            control.update_continuous_dim(key, cast(Tuple[float, float], val))
        elif key in control.discrete_dims:
            control.update_discrete_dim(key, cast(List[Any], val))
        else:
            raise AttributeError(
                f"Attribute {key} unknown in {type(control).__name__}")

def init_control(cfg: Dict[str, Any], root_folder: str):
    args = cfg.get('args', {})
    module = importlib.import_module(cfg['module'])

    control_module = getattr(module, f"Control")
    control = control_module(**args, root_folder=root_folder)
    filtered_desc = {k: v for (k, v) in cfg.items() if k not in ['args', 'module']}
    overwrite_control(control, filtered_desc)
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
