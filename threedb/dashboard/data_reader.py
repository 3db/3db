"""
threedb.dashboard.data_reader
=============================

Defines the DataReader class that processes log folders
"""

import json
from os import path

import numpy as np
from tqdm import tqdm


class DataReader:
    """
    Class that reads a log folder and packs the data in order to be sent to
    the client


    """

    def __init__(self, logdir: str):
        """Initializes the reader

        Parameters
        ----------

        logdir: str
            The folder where the logs are stored
        """

        self.logdir = logdir
        self.last_size = 0
        self.next_ix = 0
        self.keys = None
        self.result = None
        self.answer = '{}'
        self.class_map = []

        self.fname = path.join(self.logdir, 'details.log')

        self.load_class_map()

    def load_class_map(self):
        try:
            with open(path.join(self.logdir, 'class_maps.json')) as handle:
                self.class_map = json.load(handle)
        except:
            raise


    def has_changed(self) -> bool:
        """
        Checks whether the logs have been updated and the outputs need
        to be regenerated

        Returns
        -------

        bool:
            True if outdated, False otherwise
        """

        current_size = path.getsize(self.fname)
        return current_size > self.last_size

    def update_data(self):
        if not self.has_changed():
            return

        # Make sure this is identical to EXTRA_KEYS in DetailView.js on the client
        result = None
        with open(self.fname) as handle:
            full_data = handle.readlines()
            handle.seek(0, 2)
            self.last_size = handle.tell()
            n_samples = len(full_data) - self.next_ix - 1
            for i in tqdm(range(n_samples)):
                json_data = full_data[self.next_ix + i]
                data = json.loads(json_data)
                all_keys = [x for x in list(data.keys()) if x != 'render_args']
                for p in data['render_args'].keys():
                    all_keys.append('render_args.' + p)

                if self.keys is None:
                    self.keys = all_keys

                if result is None:
                    result = np.zeros((n_samples, len(self.keys)), dtype='object')
                else:
                    assert self.keys == all_keys

                for kix, k in enumerate(self.keys):
                    source = data
                    if k.startswith('render_args.'):
                        k = k.replace('render_args.', '')
                        source = data['render_args']
                    s = source[k]
                    if s == "True":
                        s = True
                    elif s == "False":
                        s = False
                    result[i, kix] = s

            self.next_ix = len(full_data)
        if self.result is None:
            self.result = result
        else:
            if result is not None:
                self.result = np.concatenate([self.result, result])
        self.prepare_answer()

    def prepare_answer(self):
        self.answer = json.dumps({
            'parameters': list(self.keys),
            'data': self.result.tolist(),
            'class_map': self.class_map
        })

