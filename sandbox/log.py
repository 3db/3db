import threading
import numpy as np
from queue import Queue, Empty
import json
from torch.utils.tensorboard import SummaryWriter
from IPython import embed
import cv2
from os import path
import pandas as pd

def clean_key(k):
    if isinstance(k, str):
        return k
    return ".".join(k)

def clean_value(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, dict):
        return clean_log(v)
    else:
        return v

def clean_log(d):
    return {clean_key(k): clean_value(v) for (k, v) in d.items()}


class Logger(threading.Thread):

    def __init__(self):
        super().__init__()
        self.queue = Queue()

    def log(self, data):
        self.queue.put(data)

    def run(self):
        raise NotImplementedError


class JSONLogger(Logger):

    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        print(f'==>[Logging to the JSON file {fname}]')

    def run(self):
        with open(self.fname, 'a+') as handle:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                handle.write(json.dumps(clean_log(item)))
                handle.write('\n')


class TbLogger(Logger):

    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        print(f'==>[Loggint to tensorboard at {dir}]')
        self.writer = SummaryWriter()
        self.df = None
        self.numeric_data = []
        self.images = []
        self.count = 0

    def write(self):
        self.df = pd.DataFrame(self.numeric_data)
        
        # 
        self.writer.add_scalar('Accuracy', self.df.is_correct.mean(), self.count)

    def run(self):
        while True:
            self.count += 1
            item = self.queue.get()
            if item is None:
                break
            self.numeric_data.append({k:v for k, v in item if k!='image'})
            self.images.append({item['id']: item['image']})
            if self.count % 1 == 0:
                self.write()

class ImageLogger(Logger):

    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        print(f'==>[Logging images to {dir}]')

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            img_path = path.join(self.dir, item['id'] + '.png')
            cv2.imwrite(img_path, item['image'])