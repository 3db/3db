import zmq
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

arguments = sys.argv[1:]
try:
    index = arguments.index('--')
    arguments = arguments[index + 1:]
except ValueError:
    pass


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

    def query(kind, **args):
        socket.send_pyobj({
            'kind': kind,
            'worker_id': worker_id,
            **args
        })
        return socket.recv_pyobj()

    while True:

        infos = query('info')
        render_args = infos['render_args']

        rendering_engine = importlib.import_module(render_args['engine'])
        all_models = rendering_engine.enumerate_models(args.root_folder)
        all_envs = rendering_engine.enumerate_environments(args.root_folder)

        assert set(infos['models']) == set(all_models)
        assert set(infos['environments']) == set(all_envs)

        assignment = query('connect')

        assert assignment['kind'] == 'assignment'
        uid_to_logits = assignment['uid_to_logits']
        inference_args = assignment['inference']
        inference_model = load_inference_model(inference_args)

        rendering_engine.load_env(args.root_folder, assignment['environment'])
        loaded_model = rendering_engine.load_model(args.root_folder,
                                                   assignment['model'])

        model_uid = rendering_engine.get_model_uid(loaded_model)

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
                                                     controls_applier)
                    result = controls_applier.apply_post_controls(result)
                    result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)

                    prediction = inference_model(result)
                    is_correct = prediction.argmax() in uid_to_logits[model_uid]
                    query('push', job=job, result=(result, prediction, is_correct))
            print(job_description)
