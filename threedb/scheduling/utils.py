"""
Scheduling utils
================
"""

from threedb.utils import CyclicBuffer
import zmq
import numpy as np
import torch as ch
from typing import Dict, Any

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
            cyclic_buffer: CyclicBuffer) -> Dict[str, Any]:
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