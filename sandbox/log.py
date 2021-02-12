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

class LoggerManager():

    def __init__(self):
        super().__init__()
        self.loggers = []

    def append(self, logger):
        self.loggers.append(logger)

    def log(self, item):
        for logger in self.loggers:
            logger.enqueue(item)

    def start(self):
        for logger in self.loggers:
            logger.start()

    def join(self):
        for logger in self.loggers:
            logger.join()


class Logger(Process):

    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def enqueue(self, item):
        self.queue.put(item)

    def log(self):
        raise NotImplementedError()

    def end(self):
        pass

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            self.log(item)
        self.end()


class JSONLogger(Logger):

    def __init__(self, fname, result_buffer, config):
        super().__init__()
        self.handle = open(fname, 'ab+')
        self.fname = fname
        self.config = config
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
        self.queue = Queue()
        print(f'==>[Logging to the JSON file {fname} with regid {self.regid}]')

    def log(self, item):
        item = {k: v for (k, v) in item.items()}
        rix = item['result_ix']
        _, outputs, is_correct = self.result_buffer[rix]
        item['outputs'] = outputs.numpy()
        item['is_correct'] = is_correct
        item['eval_module'] = self.config['evaluation']['module']
        cleaned = clean_log(item)
        encoded = json.dumps(cleaned, default=default, option=json.OPT_SERIALIZE_NUMPY | json.OPT_APPEND_NEWLINE)
        self.result_buffer.free(rix, self.regid)
        self.handle.write(encoded)

    def end(self):
        self.handle.close()



class TbLogger(Logger):

    def __init__(self, dir, result_buffer, config):
        super().__init__()
        self.dir = dir
        print(f'==>[Logging tensorboard to {dir}]')
        self.writer = None # Defer allocation in the sub-process
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
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
        self.images[item['id']] = image['rgb'].clone()

        if self.count % 1 == 0:
            self.write()

        self.result_buffer.free(rix, self.regid)


class ImageLogger(Logger):

    def __init__(self, dir, result_buffer, config):
        super().__init__()
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
        self.folder = dir
        print(f'==>[Logging images to {dir} with regid {self.regid}]')

    def log(self, item):
        rix = item['result_ix']
        images, _, __ = self.result_buffer[rix]
        for channel_name, image in images.items():
            if channel_name == 'segmentation':
                img_path = path.join(self.folder,
                                     item['id'] + '_' + channel_name + '.npy')
                np.save(img_path, image.numpy()[0])
            else:
                img_path = path.join(self.folder,
                                     item['id'] + '_' + channel_name + '.png')
                cv2.imwrite(img_path, cv2.cvtColor(image.permute(1,2,0).numpy()*255.0, cv2.COLOR_RGB2BGR))
        self.result_buffer.free(rix, self.regid)
