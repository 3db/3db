import threading
import numpy as np
from queue import Queue, Empty
import json
from torch.utils.tensorboard import SummaryWriter
import cv2
from os import path
import pandas as pd
import torch as ch
import torchvision


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
    return {clean_key(k): clean_value(v) for (k, v) in d.items() if k!='image'}

class LoggerManager(threading.Thread):
    
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

    def __init__(self, fname):
        self.fname = fname
        print(f'==>[Logging to the JSON file {fname}]')

    def log(self, item):
        with open(self.fname, 'a+') as handle:
            handle.write(json.dumps(clean_log(item)))
            handle.write('\n')


class TbLogger():

    def __init__(self, dir):
        self.dir = dir
        print(f'==>[Loggint tensorboard to {dir}]')
        self.writer = SummaryWriter(log_dir=dir)
        self.numeric_data = []
        self.images = {}
        self.count = 0

    def write(self):
        df = pd.DataFrame(self.numeric_data)
        self.writer.add_scalar('Accuracy', df.is_correct.mean(), self.count)
        for uid in df.model.unique():
            id = df[df.model == uid].id.sample(1).item()
            self.writer.add_image(uid, self.images[id], self.count)

    def log(self, item):
        self.count += 1
        self.numeric_data.append({k: v for k, v in item.items() if k!='image'})
        self.images[item['id']] = item['image']
        if self.count % 1 == 0:
            self.write()

class ImageLogger():

    def __init__(self, dir):
        self.dir = dir
        print(f'==>[Logging images to {dir}]')

    def log(self, item):
        img_path = path.join(self.dir, item['id'] + '.png')
        cv2.imwrite(img_path, cv2.cvtColor(item['image'], cv2.COLOR_RGB2BGR)) 
