import zmq
from tqdm import tqdm
import numpy as np
import torch as ch
from uuid import uuid4
import sys
import time
from os import path
import importlib
import cv2
import argparse
import sandbox
from glob import glob

from sandbox.utils import load_inference_model
from sandbox.rendering.utils import ControlsApplier
from types import SimpleNamespace

arguments = sys.argv[1:]
try:
    index = arguments.index('--')
    arguments = arguments[index + 1:]
except ValueError:
    pass

COUNTER = 0

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    if ch.is_tensor(A):
        A = A.data.cpu().numpy()
    A = np.ascontiguousarray(A)
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


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
                        default=256, type=int)

    args = parser.parse_args(arguments)
    print(args)

    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://" + args.master_address)

    worker_id = str(uuid4())

    def query(kind, result=None, **args):
        to_send = {
            'kind': kind,
            'worker_id': worker_id,
            **args
        }

        if result is not None:
            to_send['result'] = True

        socket.send_json(to_send, flags=zmq.SNDMORE if result is not None else 0)

        if result is not None:
            image, logits, is_correct = result
            send_array(socket, image, flags=zmq.SNDMORE)
            send_array(socket, logits, flags=zmq.SNDMORE)
            socket.send_pyobj(is_correct)

        result = socket.recv_pyobj()
        if result['kind'] == 'die':
            print("==>[Received closed request from master]")
            exit()
        return result

    while True:
        infos = query('info')
        render_args = infos['render_args']

        rendering_engine = importlib.import_module(render_args['engine'])
        all_models = rendering_engine.enumerate_models(args.root_folder)
        all_envs = rendering_engine.enumerate_environments(args.root_folder)

        assert set(infos['models']) == set(all_models)
        assert set(infos['environments']) == set(all_envs)

        print("INFO gotten")
        assignment = query('connect')

        assert assignment['kind'] == 'assignment'
        uid_to_logits = assignment['uid_to_logits']
        inference_args = assignment['inference']
        inference_model = load_inference_model(inference_args)

        loaded_env = rendering_engine.load_env(args.root_folder,
                                             assignment['environment'])
        loaded_model = rendering_engine.load_model(args.root_folder,
                                                   assignment['model'])

        model_uid = rendering_engine.get_model_uid(loaded_model)
        renderer_settings = SimpleNamespace(**vars(args),
                                            **render_args)
        rendering_engine.setup_render(renderer_settings)

        pbar = tqdm(smoothing=0)

        while True:
            print("starting to pull")
            job_description = query('pull', batch_size=120)
            if job_description['kind'] == 'done':  # Configuration is done, reconnect
                print("This configuration is done")
                break
            print("pull done")
            paramters = job_description['params_to_render']
            controls_args = job_description['controls_args']
            if len(paramters) == 0:
                print("Nothing to do!", 'sleeping')
                time.sleep(1)
            else:
                print("do some work")
                for job in paramters:
                    controls_applier = ControlsApplier(job.control_order,
                                                       job.render_args,
                                                       controls_args,
                                                       args.root_folder)

                    result = rendering_engine.render(model_uid, job, args,
                                                     render_args,
                                                     controls_applier,
                                                     loaded_model,
                                                     loaded_env
                                                     )
                    result = controls_applier.apply_post_controls(result)
                    result = result[:3]

                    prediction = inference_model(result)
                    is_correct = prediction.argmax() in uid_to_logits[model_uid]
                    data = (result, prediction, is_correct)
                    query('push', job=job.id, result=data)
                    pbar.update(1)
            # print(job_description)
