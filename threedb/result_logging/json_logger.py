"""
threedb.result_logging.json_logger
==================================

Subclass of :mod:`threedb.result_logging.base_logger.BaseLogger`.
"""

import copy
import importlib
import shutil
from os import path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import orjson as json
import torch as ch
from threedb.result_logging.base_logger import BaseLogger
from threedb.utils import CyclicBuffer

def clean_key(k: Iterable[str]) -> str:
    """
    Utility function for formatting keys.
    This is a no-op if the input is a string, otherwise expects an iterable
    of strings, which it joins with a period.
    """
    return k if isinstance(k, str) else '.'.join(k)

def clean_value(val: Any):
    """
    Utility function for formatting tensors.
    Converts torch tensors to numpy, is a no-op for all other types.
    """
    return val.numpy() if ch.is_tensor(val) else val

def json_default(obj: np.ndarray) -> str:
    """
    Another utility function; turns floats into strings, otherwise (if the
    input does not have type ``np.float64`` raises a ``TypeError``.
    """
    if isinstance(obj, np.ndarray):
        return str(obj)
    raise TypeError

def clean_log(log_d: dict, key_blacklist: Tuple[str, str] = ('image', 'result_ix')) -> dict:
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


class JSONLogger(BaseLogger):
    def __init__(self,
                 root_dir: str,
                 result_buffer: CyclicBuffer,
                 config: Dict[str, Dict[str, Any]]) -> None:
        """
        A logger that logs all an experiments meta-data and results into a JSON file.
        """
        super().__init__(root_dir, result_buffer, config)
        fname = path.join(root_dir, 'details.log')
        self.handle = open(fname, 'ab+')
        self.regid = self.buffer.register()
        self.evaluator = importlib.import_module(self.config['evaluation']['module']).Evaluator
        if 'label_map' in config['inference']:
            classmap_fname = path.join(root_dir, 'class_maps.json')
            print(f"==> [Saving class maps to {classmap_fname}]")
            shutil.copyfile(config['inference']['label_map'], classmap_fname)
        print(f'==> [Logging to the JSON file {fname} with regid {self.regid}]')

    def log(self, item: Dict[str, Any]) -> None:
        """Concrete implementation of
        :meth:`threedb.result_logging.base_logger.BaseLogger.log`.

        Parameters
        ----------
        item : Dict[str, Any]
            The item to be logged.
        """
        item = copy.deepcopy(item)
        rix = item['result_ix']
        buffer_data = self.buffer[rix]
        result = {k: v for (k, v) in buffer_data.items() if k in self.evaluator.KEYS}
        for k in ['id', 'environment', 'model', 'render_args']:
            result[k] = item[k]
        result['output_type'] = self.evaluator.output_type
        cleaned = clean_log(result)
        encoded = json.dumps(cleaned, default=json_default,
                             option=json.OPT_SERIALIZE_NUMPY | json.OPT_APPEND_NEWLINE)
        self.buffer.free(rix, self.regid)
        self.handle.write(encoded)

    def end(self):
        """Concrete implementation of
        :meth:`threedb.result_logging.base_logger.BaseLogger.end`. 
        
        Closes the necessary file handle.
        """
        self.handle.close()

Logger = JSONLogger
