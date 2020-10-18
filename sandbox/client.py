import zmq
from uuid import uuid4
import sys
import time
from os import path
import argparse
import sandbox
from glob import glob


from sandbox.rendering.render import render, load_model, load_env

from sandbox.utils import load_inference_model


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
        all_envs = [path.basename(x) for x in glob(path.join(args.root_folder, 'environments', '*.blend'))]
        all_models = [path.basename(x) for x in glob(path.join(args.root_folder, '3Dmodels', '*.blend'))]

        infos = query('info')
        assert set(infos['models']) == set(all_models)
        assert set(infos['environments']) == set(all_envs)

        assignment = query('connect')

        assert assignment['kind'] == 'assignment'
        env = path.join(args.root_folder, 'environments', assignment['environment'])
        model = path.join(args.root_folder, '3Dmodels', assignment['model'])
        uid_to_logits = assignment['uid_to_logits']
        inference_args = assignment['inference']
        inference_model = load_inference_model(inference_args)

        load_env(env)
        model_uid = load_model(model)

        while True:
            print("starting to pull")
            job_description = query('pull', batch_size=120)
            if job_description['kind'] == 'done':  # Configuration is done, reconnect
                print("This configuration is done")
                break
            print("pull done")
            paramters = job_description['params_to_render']
            render_args = job_description['render_args']
            controls_args = job_description['controls_args']
            if len(paramters) == 0:
                print("Nothing to do!", 'sleeping')
                time.sleep(1)
            else:
                print("do some work")
                for job in paramters:
                    result = render(model_uid, job, args, render_args, controls_args)
                    prediction = inference_model(result)
                    is_correct = prediction.argmax() in uid_to_logits[model_uid]
                    query('push', job=job, result=(result, prediction, is_correct))
            print(job_description)

    print(env, model)



