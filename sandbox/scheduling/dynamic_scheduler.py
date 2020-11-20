import zmq
import numpy as np
import random
from tqdm import tqdm
import torch as ch
import json

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    A = A.reshape(md['shape'])
    A = ch.from_numpy(A.copy())
    return A

def my_recv(socket, cyclic_buffer):
    main_message = socket.recv_json()

    if 'result' in main_message:
        image = recv_array(socket)
        logits = recv_array(socket)
        is_correct = socket.recv_pyobj()

        ix = cyclic_buffer.allocate(image, logits, is_correct)
        main_message['result'] = ix

    return main_message

def schedule_work(policy_controllers, port, list_envs, list_models,
                  render_args, inference_args, controls_args,
                  result_buffer):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    workers = {}

    controllers_to_start = policy_controllers[:]
    running_policies = set()
    done_policies = set()

    # Load the mapping for UIDs to logits indices
    with open(inference_args['uid_to_logits'], 'r') as f:
        uid_to_logits = json.load(f)


    rendering_bar = tqdm(smoothing=0.1, unit=' images')
    rendering_bar.set_description('Rendering')
    policies_bar = tqdm(total=len(policy_controllers), unit=' policies')
    policies_bar.set_description('Policies')

    while True:
        if len(done_policies) == len(policy_controllers):
            break  # We finished all the policies

        message = my_recv(socket, result_buffer)

        wid = message['worker_id']

        if message['kind'] == 'info':
            socket.send_pyobj({
                'kind': 'info',
                'environments': list_envs,
                'models': list_models,
                'render_args': render_args,
            })
        elif message['kind'] == 'connect':
            if len(controllers_to_start):
                selected_policy = controllers_to_start.pop()
                selected_policy.start()
                running_policies.add(selected_policy)
            else:
                selected_policy = random.choice(list(running_policies))

            assert(wid not in workers)
            workers[wid] = selected_policy
            socket.send_pyobj({
                'kind': 'assignment',
                'environment': selected_policy.env_file,
                'model': selected_policy.model_name,
                'uid_to_logits': uid_to_logits,
                'inference': inference_args
            })

        elif message['kind'] == 'pull':
            selected_policy = workers[wid]
            if selected_policy in done_policies:

                del workers[wid]
                socket.send_pyobj({
                    'kind': 'done'
                })
            else:
                bs = message['batch_size']
                result = []
                for _ in range(bs):
                    pulled = selected_policy.pull_work(wid)
                    if pulled is None:
                        break
                    result.append(pulled)

                socket.send_pyobj({
                    'kind': 'work',
                    'environment': selected_policy.env_file,
                    'model': selected_policy.model_name,
                    'params_to_render': result,
                    'controls_args': controls_args
                })

        elif message['kind'] == 'push':
            selected_policy = workers[wid]
            selected_policy.push_result(message['job'], message['result'])
            socket.send_pyobj({
                'kind': 'ack'
            })
            rendering_bar.update(1)
            rendering_bar.set_postfix({
                'workers': len(workers),
            })

        else:
            socket.send_pyobj({
                'kind': 'bad_query'
            })

        for policy in list(running_policies):
            if not policy.is_alive():
                policies_bar.update(1)
                policies_bar.set_postfix({
                    'concurrent running': len(running_policies)
                })
                running_policies.remove(policy)
                done_policies.add(policy)

    rendering_bar.close()
    policies_bar.close()

    print("==>[Received all the results]")
    print("==>[Shutting down workers]")
    for _ in tqdm(range(len(workers)), desc='Shutting down', unit=' workers'):
        message = my_recv(socket, result_buffer)
        socket.send_pyobj({
            'kind': 'die'
        })
        try:
            wid = message['worker_id']
            del workers[wid]
        except:  # This worker didn't do work yet, we still shut it down
            pass
