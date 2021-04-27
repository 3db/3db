"""
threedb.client
==============

The client is responsible for receiving tasks (sets of parameters) from the
main node, rendering these tasks, performing inference, and then returning
the results to the main node for inference.

Users should not have to modify or even view this file in order to use and
extend 3DB.
"""
import argparse
import importlib
import sys
import time
from types import SimpleNamespace
from typing import Any, Optional, Type, Dict
from uuid import uuid4

import numpy as np
import torch as ch
import zmq
from tqdm import tqdm

from threedb.rendering.utils import ControlsApplier
from threedb.rendering.base_renderer import BaseRenderer
from threedb.evaluators.base_evaluator import BaseEvaluator
from threedb.utils import load_inference_model

COUNTER = 0

def send_array(sock: zmq.Socket,
               arr: Any,
               dtype: Optional[str]=None,
               flags: int=0,
               copy: bool=True,
               track: bool=False):
    """send a numpy array with metadata"""
    if ch.is_tensor(arr):
        arr = arr.data.cpu().numpy()
    arr = np.ascontiguousarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    message_dict = dict(
        dtype=str(arr.dtype),
        shape=arr.shape,
    )
    sock.send_json(message_dict, flags | zmq.SNDMORE)
    return sock.send(arr, flags, copy=copy, track=track)

def query(sock: zmq.Socket, kind: str, worker_id: str, 
          result_data: Optional[Dict[str, Any]] = None,
          result_dtypes: Optional[Dict[str, str]] = None,
          **kwargs) -> Dict[str, Any]:
    """Send a request back to the server and receive a response. Additional
    named arguments are forwarded to the server as-is as part of the request.

    Parameters
    ----------
    sock : zmq.Socket
        An open socket connected to the same port as the server.
    kind : str
        What kind of request to send (``'info'``, ``'push'``, ``'pull'``, or
        ``'decl'``)
    worker_id : str
        The id of the client sending the request
    result_data : Optional[Dict[str, Any]], optional
        If ``kind == 'push'``, this should be a dictionary of results to send
        back to the server (otherwise ignored), by default None.

    Returns
    -------
    Dict[str, Any]
        The response from the server for to the sent message; see `here <extending.html#the-3db-workflow>`_ for
        documentation of the communication protocol,
    """            
    to_send = {
        'kind': kind,
        'worker_id': worker_id,
        **kwargs
    }

    if result_data is not None:
        result_keys = list(result_data.keys())
        to_send['result_keys'] = result_keys

        sock.send_json(to_send, flags=zmq.SNDMORE)
        for channel_name in result_keys:
            send_array(sock,
                       result_data[channel_name],
                       result_dtypes.get(channel_name, None),
                       flags=zmq.SNDMORE)
        sock.send_string('done')
    else:
        sock.send_json(to_send, flags=0)

    response = sock.recv_pyobj()
    if kind == 'decl':
        assert response['kind'] == 'ack', 'Received a non-ack message from the server, abort.'
        
    if response['kind'] == 'die':
        print("==> [Received closed request from master]")
        sys.exit()
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Render worker for the robustness threedb')
    parser.add_argument('root_folder', type=str,
                        help='folder containing all data (models, environments, etc)')
    parser.add_argument('--master-address', '-a', type=str,
                        help='How to contact the master node',
                        default='localhost:5555')
    parser.add_argument('--gpu-id', help='The GPU to use to render (-1 for cpu)',
                        default=-1, type=int,)
    parser.add_argument('--cpu-cores', help='number of cpu cores to use (default uses all)',
                        default=None, type=int,)
    parser.add_argument('--tile-size', help='The size of tiles used for GPU rendering',
                        default=32, type=int)
    parser.add_argument('--batch-size', help='How many task to ask for in a batch',
                        default=1, type=int)
    parser.add_argument('--fake-results', action='store_true',
                        help='Always return the same result regardless of the parameters'
                             '\n useful to debug and produce large amount of data quickly')

    arguments = sys.argv[1:]
    try:
        index = arguments.index('--')
        arguments = arguments[index + 1:]
    except ValueError:
        pass
    args = parser.parse_args(arguments)
    print(args)

    context = zmq.Context()
    print(f"Connecting to server ({args.master_address})...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://" + args.master_address)

    WORKER_ID = str(uuid4())
    LAST_RESULT = []  # This is used to store the first render when --fake-result is set
    # while True:
    infos = query(socket, 'info', WORKER_ID)
    render_args = infos['render_args']

    rendering_class: Type[BaseRenderer] = getattr(importlib.import_module(render_args['engine']), 'Renderer')
    rendering_engine: BaseRenderer = rendering_class(args.root_folder, {**render_args, **vars(args)})

    evaluation_args = infos['evaluation_args']
    evaluator_class = getattr(importlib.import_module(evaluation_args['module']), 'Evaluator')
    evaluator: BaseEvaluator = evaluator_class(**infos['evaluation_args']['args'])

    # Gather all experiment-wide parameters
    inference_args = infos['inference']
    controls_args = infos['controls_args']
    inference_model = load_inference_model(inference_args)

    last_env = None
    last_model = None

    pbar = tqdm(smoothing=0)

    image_shapes = rendering_engine.declare_outputs()
    assert set(image_shapes.keys()).issubset(rendering_class.KEYS), \
        'Return value of declare_outputs() should match the declared KEYS var'
    eval_shapes = evaluator.declare_outputs()
    assert set(eval_shapes.keys()).issubset(evaluator_class.KEYS), \
        'Return value of declare_outputs() should match the declared KEYS var'
    declared_outputs = {
        **image_shapes,
        **eval_shapes,
        'output': (inference_args['output_shape'], 'float32'),
    }
    query(socket, 'decl', WORKER_ID, declared_outputs=declared_outputs)

    # These will be assigned in the if statement below
    model_uid = ''
    while True:
        job_description = query(socket, 'pull', WORKER_ID,
                                batch_size=args.batch_size,
                                last_environment=last_env,
                                last_model=last_model)
        parameters = job_description['params_to_render']

        if len(parameters) == 0:
            print("Nothing to do!", 'sleeping')
            time.sleep(1)
            continue

        for job in parameters:
            if LAST_RESULT:
                data = LAST_RESULT[0]
            else:
                current_env = job.environment
                current_model = job.model

                # We reload model and env if we got assigned to something
                # different this time
                if current_env != last_env or current_model != last_model:
                    print("==> [Loading new environment/model pair]")
                    loaded_env = rendering_engine.load_env(current_env)
                    loaded_model = rendering_engine.load_model(current_model)
                    model_uid = rendering_engine.get_model_uid(loaded_model)
                    rendering_engine.setup_render(loaded_model, loaded_env)
                    last_env = current_env
                    last_model = current_model

                controls_applier = ControlsApplier(job.control_order,
                                                job.render_args,
                                                controls_args,
                                                args.root_folder)

                scalar_label = evaluator.get_segmentation_label(model_uid)
                render_context = rendering_engine.get_context_dict(model_uid, scalar_label)
                controls_applier.apply_pre_controls(render_context)
                result = rendering_engine.render(model_uid,
                                                 loaded_model,
                                                 loaded_env)
                result['rgb'] = controls_applier.apply_post_controls(result['rgb'])
                controls_applier.unapply(render_context)

                with ch.no_grad():
                    result['rgb'] = result['rgb'][:3]
                    prediction, input_shape = inference_model(result['rgb'])

                lab = evaluator.get_target(model_uid, result)
                evaluation = evaluator.summary_stats(prediction, lab)
                assert evaluation.keys() == eval_shapes.keys(), \
                    'Outputs do not match declared outputs' \
                   f'{list(evaluation.keys())}, {list(eval_shapes.keys())}'
                out_shape =  inference_args['output_shape']
                prediction_tens = evaluator.to_tensor(prediction, out_shape, input_shape)
                data = {
                    'output': prediction_tens,
                    **result,
                    **evaluation
                }

                if args.fake_results:
                    LAST_RESULT.append(data)

            result_dtypes = {k: v[1] for (k, v) in declared_outputs.items()}
            query(socket, 'push', WORKER_ID, result_data=data,
                  result_dtypes=result_dtypes, job=job.id)
            pbar.update(1)
