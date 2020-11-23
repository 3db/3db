import threading
from multiprocessing import cpu_count
import numpy as np
import orjson as json
import torch
from multiprocessing import Pool
import multiprocessing
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor
import cv2
from os import path
import pandas as pd
import torch as ch
import torchvision
from multiprocessing import Process, Queue

# torch.multiprocessing.set_sharing_strategy('file_system')


def clean_key(k):
    if isinstance(k, str):
        return k
    return ".".join(k)

def default(obj):
    if isinstance(obj, np.float64):
        return str(obj)
    raise TypeError

def clean_value(v):
    if ch.is_tensor(v):
        return v.numpy()
    elif isinstance(v, dict):
        return clean_log(v)
    else:
        return v

def clean_log(d):
    return {clean_key(k): clean_value(v) for (k, v) in d.items() if k!='image' and k != 'result_ix'}

class LoggerManager(Process):

    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.loggers = []

    def append(self, logger):
        self.loggers.append(logger)

    def log(self, data):
        self.queue.put(data)

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            for logger in self.loggers:
                logger.log(item)



class JSONLogger():

    def __init__(self, fname, result_buffer):
        self.handle = open(fname , 'ab+')
        self.fname = fname
        self.result_buffer = result_buffer
        self.result_buffer.register()
        print(f'==>[Logging to the JSON file {fname}]')

    def log(self, item):
        item = {k:v for (k,v) in item.items()}
        rix = item['result_ix']
        _, logits, is_correct = self.result_buffer[rix]
        item['logits'] = logits.numpy()
        item['is_correct'] = is_correct
        cleaned = clean_log(item)
        encoded = json.dumps(cleaned, default=default, option=json.OPT_SERIALIZE_NUMPY | json.OPT_APPEND_NEWLINE)
        self.result_buffer.free(rix)
        self.handle.write(encoded)


class TbLogger():

    def __init__(self, dir, result_buffer):
        self.dir = dir
        print(f'==>[Loggint tensorboard to {dir}]')
        self.writer = None # Defer allocation in the sub-process
        self.result_buffer = result_buffer
        self.result_buffer.register()
        self.numeric_data = []
        self.images = {}
        self.count = 0

    def write(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.dir)
        df = pd.DataFrame(self.numeric_data)
        current_acc = df.is_correct.mean()
        self.writer.add_scalar('Accuracy', current_acc, self.count)
        for uid in df.model.unique():
            id = df[df.model == uid].id.iloc[-1]
            self.writer.add_image(uid, self.images[id], self.count)
        self.images = {}
        self.numeric_data = []

    def log(self, item):
        self.count += 1
        rix = item['result_ix']
        image, __, is_correct = self.result_buffer[rix]
        information = {k: v for k, v in item.items() if k != 'result_ix'}
        information['is_correct'] = is_correct

        self.numeric_data.append(information)
        self.images[item['id']] = image.clone()

        if self.count % 1 == 0:
            self.write()

        self.result_buffer.free(rix)


class ImageLogger():

    def __init__(self, dir, result_buffer):
        self.result_buffer = result_buffer
        self.result_buffer.register()
        self.folder = dir
        print(f'==>[Logging images to {dir}]')

    def log(self, item):
        rix = item['result_ix']
        image, _, __ = self.result_buffer[rix]
        img_path = path.join(self.folder, item['id'] + '.png')
        cv2.imwrite(img_path, cv2.cvtColor(image.permute(1,2,0).numpy()*255.0, cv2.COLOR_RGB2BGR))
        self.result_buffer.free(rix)
