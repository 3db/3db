import threading
import numpy as np
from queue import Queue, Empty
import json

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

    def __init__(self, fname):
        super().__init__()
        self.queue = Queue()
        self.fname = fname

    def log(self, data):
        self.queue.put(data)

    def run(self):
        with open(self.fname, 'a+') as handle:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                handle.write(json.dumps(clean_log(item)))
                handle.write('\n')


