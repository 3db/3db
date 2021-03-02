import copy
import importlib
import shutil
from multiprocessing import Process, Queue
from os import path
from typing import Iterable, Tuple
from sandbox.utils import BigChungusCyclicBuffer

import cv2
import numpy as np
import orjson as json
import pandas as pd
import torch as ch
from torch.utils.tensorboard import SummaryWriter


def clean_key(k: Iterable[str]) -> str:
    """
    Utility function for formatting keys.
    This is a no-op if the input is a string, otherwise expects an iterable
    of strings, which it joins with a period.
    """
    return k if isinstance(k, str) else '.'.join(k)

def clean_value(val: object):
    """
    Utility function for formatting tensors.
    Converts torch tensors to numpy, is a no-op for all other types.
    """
    return val.numpy() if ch.is_tensor(val) else val

def default(obj: np.ndarray) -> str:
    """
    Another utility function; turns floats into strings, otherwise (if the
    input does not have type ``np.float64`` raises a ``TypeError``.
    """
    if isinstance(obj, np.ndarray):
        return str(obj)
    raise TypeError

def clean_log(log_d: dict, key_blacklist: Tuple[str] = ('image', 'result_ix')) -> dict:
    """
    Cleans a dictionary for log-writing. In particular, all keys (expected to
    be either strings or iterables of strings) are converted to strings, and
    all torch tensors are converted to numpy arrays.
    """
    cleaned = {}
    for k, val in log_d.items():
        if k in key_blacklist:
            continue
        clean_v = clean_log(val) if isinstance(val, dict) else clean_value(val)
        cleaned[clean_key(k)] = clean_v
    return cleaned

class Logger(Process):
    """
    Abstract class for a Logger, inherits from ``multiprocessing.Process``.
    Implements the additional functions:

    - enqueue(): put a new item on the logging queue, to be logged ASAP
    - log(): the actual logging mechanism, meant to be overwritten by the
        user for each unique subclass. This should not be called directly,
        but rather will be called by ``run()``
    - run(): main loop, waits for logs to be added to the queue, and calls
        ``log()`` on them.
    [TODO]
    """
    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def enqueue(self, item: dict):
        """
        Add an item to the queue to be logged. See the documentation of
            :meth:``[TODO]`` for the required structure of the log item.
        """
        self.queue.put(item)

    def log(self, item: dict):
        """
        Log an item.

        Arguments:
        - item (dict) : must have keys: ``outputs``, ``is_correct`` and ``output_type``:
            - ``outputs`` should be a tensor of model outputs (predictions)
            - ``outputs TODO
        """
        raise NotImplementedError

    def end(self):
        """
        Performs cleanup operations for the logger. No-op by default, should be
        overriden with code for closing any open file handles, ports, etc.
        """
        pass

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            self.log(item)
        self.end()

class JSONLogger(Logger):
    def __init__(self, root_dir: str, result_buffer: BigChungusCyclicBuffer, config: dict):
        super().__init__()
        fname = path.join(root_dir, 'details.log')
        self.handle = open(fname, 'ab+')
        self.config = config
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
        self.queue = Queue()
        self.evaluator = importlib.import_module(self.config['evaluation']['module']).Evaluator
        if 'label_map' in config['inference']:
            classmap_fname = path.join(root_dir, 'class_maps.json')
            print(f"==> [Saving class maps to {classmap_fname}]")
            shutil.copyfile(config['inference']['label_map'], classmap_fname)
        print(f'==> [Logging to the JSON file {fname} with regid {self.regid}]')

    def log(self, item):
        item = copy.deepcopy(item)
        rix = item['result_ix']
        # _, outputs, is_correct = self.result_buffer[rix]
        buffer_data = self.result_buffer[rix]
        item['output'] = buffer_data['output']
        item['is_correct'] = buffer_data['corrects']
        item['output_type'] = self.evaluator.output_type
        cleaned = clean_log(item)
        encoded = json.dumps(cleaned, default=default, 
                             option=json.OPT_SERIALIZE_NUMPY | json.OPT_APPEND_NEWLINE)
        self.result_buffer.free(rix, self.regid)
        self.handle.write(encoded)

    def end(self):
        self.handle.close()

class TbLogger(Logger):
    """
    TensorBoard Logger.

    Logs accuracy (as a scalar) and all generated images to TensorBoard.
    """
    def __init__(self, tb_dir: str, result_buffer: BigChungusCyclicBuffer, _):
        super().__init__()
        self.tb_dir = tb_dir
        print(f'==> [Logging tensorboard to {tb_dir}]')
        self.writer = None # Defer allocation in the sub-process
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
        self.numeric_data = []
        self.images = {}
        self.count = 0

    def _write(self):
        """
        Internal function for writing to the tensorboard.
        """
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)
        data_df = pd.DataFrame(self.numeric_data)
        current_acc = data_df.is_correct.mean()
        self.writer.add_scalar('Accuracy', current_acc, self.count)
        for uid in data_df.model.unique():
            image_id = data_df[data_df.model == uid].id.iloc[-1]
            self.writer.add_image(uid, self.images[image_id], self.count)
        self.images = {}
        self.numeric_data = []

    def log(self, item):
        self.count += 1
        rix = item['result_ix']
        buf_data = self.result_buffer[rix]
        print(buf_data.keys(), item.keys())
        information = {k: v for k, v in item.items() if k != 'result_ix'}
        information['is_correct'] = buf_data

        self.numeric_data.append(information)
        self.images[item['id']] = image['rgb'].clone()

        if self.count % 1 == 0:
            self._write()

        self.result_buffer.free(rix, self.regid)

class ImageLogger(Logger):
    """
    This logger [TODO]
    """
    def __init__(self, save_dir, result_buffer, _):
        super().__init__()
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
        self.folder = save_dir
        print(f'==> [Logging images to {dir} with regid {self.regid}]')

    def log(self, item):
        rix = item['result_ix']
        buf_data = self.result_buffer[rix]
        for channel_name, image in images.items():
            if channel_name == 'segmentation':
                img_path = path.join(self.folder,
                                     item['id'] + '_' + channel_name + '.npy')
                np.save(img_path, image.numpy()[0])
            else:
                img_path = path.join(self.folder,
                                     item['id'] + '_' + channel_name + '.png')
                img_to_write = cv2.cvtColor(image.permute(1,2,0).numpy()*255.0, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, img_to_write)
        self.result_buffer.free(rix, self.regid)


class LoggerManager():
    """
    A LoggerManager allows us to log from several loggers at once, without
    handling each one individually.

    Users should not have to modify or subclass this to extend 3DB.
    """
    def __init__(self):
        super().__init__()
        self.loggers = []

    def append(self, logger: Logger):
        """
        Adds a new logger.
        """
        self.loggers.append(logger)

    def log(self, item):
        """
        Logs the given items from each logger under management.
        """
        for logger in self.loggers:
            logger.enqueue(item)

    def start(self):
        """
        Starts each logger under management.
        """
        for logger in self.loggers:
            logger.start()

    def join(self):
        """
        Wait for each logger under management to complete their tasks.
        """
        for logger in self.loggers:
            logger.join()
