from sandbox.utils import BigChungusCyclicBuffer
import zmq
import numpy as np
import torch as ch
from typing import Dict, Any, List, Union
from tqdm import tqdm
import time

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    A = A.reshape(md['shape'])
    A = ch.from_numpy(A.copy())
    return A

def recv_into_buffer(socket: zmq.Socket, 
            cyclic_buffer: BigChungusCyclicBuffer) -> Dict[str, Any]:
    main_message: Dict[str, Any] = socket.recv_json()

    if cyclic_buffer.initialized and ('result_keys' in main_message):
        result_keys = main_message['result_keys']
        buf_data = {}
        for result_key in result_keys:
            buf_data[result_key] = recv_array(socket)

        # outputs = recv_array(socket)
        # is_correct = socket.recv_pyobj()
        assert socket.recv_string() == 'done', 'Did not get done message'

        # ix = cyclic_buffer.allocate(images, outputs, is_correct)
        idx = cyclic_buffer.allocate(buf_data)
        main_message['result'] = idx

    return main_message

class MultiBar:
    def __init__(self, titles: List[str],
                       units: Union[List[str], str] = "",
                       smoothing: Union[List[float], float] = 1.0,
                       update_freq: float = 0.1):

        if isinstance(smoothing, float):
            smoothing = [smoothing] * len(titles)
        if isinstance(units, str):
            units = [units] * len(titles)

        print(titles, units, smoothing)
        self.update_freq = update_freq
        self.last_update = time.time()
        self.bars = {t: tqdm(unit=u, smoothing=s, total=0) for (u, t, s) in zip(units, titles, smoothing)}
        for k in self.bars:
            self.bars[k].set_description(k)

    def update_bars(self):
        if time.time() > self.last_update + self.update_freq:
            self.last_update = time.time()
            rendering_bar.update(renders_to_report)
            renders_to_report = 0
            buffer_usage_bar.reset()
            buffer_usage_bar.update(len(result_buffer.free_idx))
            buffer_usage_bar.refresh()
            policies_bar.set_postfix({
                'concurrent running': len(running_policies)
            })
            rendering_bar.set_postfix({
                'workers': len(seen_workers),
                'pending': len(work_queue),
                'waste%': (1 - valid_renders / total_renders) * 100
            })