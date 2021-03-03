"""
3DB Client

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
from typing import Optional, Type, cast
from uuid import uuid4

import numpy as np
import torch as ch
import zmq
from tqdm import tqdm

from sandbox.rendering.utils import ControlsApplier
from sandbox.rendering.base_renderer import BaseRenderer
from sandbox.utils import load_inference_model

arguments = sys.argv[1:]
try:
    index = arguments.index('--')
    arguments = arguments[index + 1:]
except ValueError:
    pass

COUNTER = 0

def send_array(sock, arr, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    if ch.is_tensor(arr):
        arr = arr.data.cpu().numpy()
    arr = np.ascontiguousarray(arr)
    message_dict = dict(
        dtype=str(arr.dtype),
        shape=arr.shape,
    )
    sock.send_json(message_dict, flags | zmq.SNDMORE)
    return sock.send(arr, flags, copy=copy, track=track)

def query(sock: zmq.Socket, kind: str, worker_id: str, result_data: Optional[dict] = None, **kwargs):
    """
    Send a request back to the server and receive a response. Additional
    named arguments are forwarded to the server as-is as part of the request.

    Arguments:
    - kind (str): what kind of request to send (``'info'``, ``'push'``,
      ``'pull'``, or ``'startup'``)
    - worker_id (str): the client id sending the request
    - result_data (dict or None): if ``kind == 'push'``, this should be a
        dictionary of results (e.g. ``{'images': images, 'outputs': outputs,
        'corrects': corrects}``) to send back to the server.
    Returns:
    - response (dict): the response from the server for to the sent message
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
            send_array(sock, result_data[channel_name], flags=zmq.SNDMORE)
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
        description='Render worker for the robustness sandbox')
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

    args = parser.parse_args(arguments)
    print(args)

    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://" + args.master_address)

    WORKER_ID = str(uuid4())
    LAST_RESULT = []  # This is used to store the first render when --fake-result is set
    # while True:
    infos = query(socket, 'info', WORKER_ID)
    render_args = infos['render_args']

    rendering_module: Type[BaseRenderer] = getattr(importlib.import_module(render_args['engine']), 'Renderer')
    rendering_engine: BaseRenderer = rendering_module(args.root_folder, {**render_args, **vars(args)})

    evaluation_args = infos['evaluation_args']
    evaluator_module = importlib.import_module(evaluation_args['module'])
    evaluator = evaluator_module.Evaluator(**infos['evaluation_args']['args'])

    # Gather all experiment-wide parameters
    uid_to_targets = infos['uid_to_targets']
    inference_args = infos['inference']
    controls_args = infos['controls_args']
    inference_model = load_inference_model(inference_args)

    last_env = None
    last_model = None

    pbar = tqdm(smoothing=0)

    image_shapes = rendering_engine.declare_outputs()
    declared_outputs = {
        **image_shapes,
        'output': (inference_args['output_shape'], 'float32'),
        'is_correct': ([], 'bool')
    }
    query(socket, 'decl', WORKER_ID, declared_outputs=declared_outputs)

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

        print("do some work")
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
                    renderer_settings = SimpleNamespace(**vars(args),
                                                        **render_args)
                    rendering_engine.setup_render(loaded_model, loaded_env)
                    # rendering_engine.setup_render(renderer_settings)
                    last_env = current_env
                    last_model = current_model

                controls_applier = ControlsApplier(job.control_order,
                                                job.render_args,
                                                controls_args,
                                                args.root_folder)

                # context = {}
                result = rendering_engine.render_and_apply(model_uid,
                                                           uid_to_targets[model_uid][0],
                                                           controls_applier,
                                                           loaded_model,
                                                           loaded_env)
                """
                result = rendering_engine.render(context,
                                                    model_uid,
                                                    uid_to_targets[model_uid][0],
                                                    job, args,
                                                    render_args,
                                                    controls_applier,
                                                    loaded_model,
                                                    loaded_env)
                """

                with ch.no_grad():
                    prediction, input_shape = inference_model(result['rgb'])

                if evaluator_module.Evaluator.label_type == 'classes':
                    lab = uid_to_targets[model_uid]
                elif evaluator_module.Evaluator.label_type == 'segmentation_map':
                    lab = result['segmentation']
                else:
                    err_msg = f'Label type {evaluator_module.Evaluator.label_type} not found'
                    raise ValueError(err_msg)

                is_correct = evaluator.is_correct(prediction, lab)
                out_shape =  inference_args['output_shape']
                prediction_tens = evaluator.to_tensor(prediction, out_shape, input_shape)
                loss = evaluator.loss(prediction, lab)
                # extra_info = evaluator.extra_info
                data = {
                    **result,
                    'output': prediction_tens,
                    'is_correct': is_correct,
                    # 'loss': loss
                }
                # data = (result, prediction_tens, is_correct)
                if args.fake_results:
                    LAST_RESULT.append(data)
            query(socket, 'push', WORKER_ID, result_data=data, job=job.id)
            pbar.update(1)
