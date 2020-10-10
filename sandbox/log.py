import threading
import numpy as np
from queue import Queue, Empty
import json
from torch.utils.tensorboard import SummaryWriter
from IPython import embed
import cv2
from os import path
import pandas as pd
import torch
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
        print(f'==>[Loggint tensorboard to {dir}]')
        self.writer = SummaryWriter(log_dir=dir)
        self.numeric_data = []
        self.images = {}
        self.count = 0
        self.PIL_TO_IMAGE = torchvision.transforms.ToTensor()


    def write(self):
        df = pd.DataFrame(self.numeric_data)
        self.writer.add_scalar('Accuracy', df.is_correct.mean(), self.count)
        for uid in df.model.unique():
            id = df[df.model == uid].id.sample(1).item()
            grid = torchvision.utils.make_grid(self.PIL_TO_IMAGE(self.images[id]))
            self.writer.add_image(uid, grid, self.count)

    def run(self):
        while True:
            self.count += 1
            item = self.queue.get()
            if item is None:
                break
            self.numeric_data.append({k: v for k, v in item.items() if k!='image'})
            self.images[item['id']]= item['image']
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
            item['image'].save(img_path)
