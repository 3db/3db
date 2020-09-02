import zmq
from uuid import uuid4
import sys
import time
from os import path
import argparse
import sandbox
from glob import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Render worker for the robustness sandbox')

    parser.add_argument('environment_folder', type=str,
                        help='folder containing all the environment (.blend)')

    parser.add_argument('model_folder', type=str,
                        help='folder containing all models (.blend files)')

    parser.add_argument('--master-address', '-a', type=str,
                        help='How to contact the master node',
                        default='localhost:5555')


    args = parser.parse_args()

    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect ("tcp://" + args.master_address)

    worker_id = str(uuid4())

    def query(kind, **args):
        socket.send_pyobj({
            'kind': kind,
            'worker_id': worker_id,
            **args
        })
        return socket.recv_pyobj()

    while True:
        all_envs = glob(path.join(args.environment_folder, '*.blend'))
        all_models = glob(path.join(args.model_folder, '*.blend'))

        infos = query('info')
        assert set(infos['models']) == set(all_models)
        assert set(infos['environments']) == set(all_envs)

        assignment = query('connect')

        assert assignment['kind'] == 'assignment'
        env = assignment['environment']
        model = assignment['model']

        while True:
            print("starting to pull")
            job_description = query('pull', batch_size=120)
            if job_description['kind'] == 'done':  # Configuration is done, reconnect
                print("This configuration is done")
                break
            print("pull done")
            paramters = job_description['params_to_render']
            if len(paramters) == 0:
                print("Nothing to do!", 'sleeping')
                time.sleep(1)
            else:
                print("do some work")
                for job in paramters:
                    query('push', job=job, result=job.order)
            print(job_description)

    print(env, model)



