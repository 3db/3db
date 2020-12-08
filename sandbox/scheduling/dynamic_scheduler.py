import zmq
import itertools
import os
import time
import numpy as np
import random
from tqdm import tqdm
import torch as ch
import json

TQDM_FREQ = 0.1

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
        channel_names = main_message['result']
        images = {}
        for channel_name in channel_names:
            images[channel_name] = recv_array(socket)

        logits = recv_array(socket)
        is_correct = socket.recv_pyobj()

        ix = cyclic_buffer.allocate(images, logits, is_correct)
        main_message['result'] = ix

    return main_message

def schedule_work(policy_controllers, port, list_envs, list_models,
                  render_args, inference_args, controls_args,
                  result_buffer):
    context = zmq.Context(io_threads=os.cpu_count())
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    seen_workers = set()

    controllers_to_start = policy_controllers[:]
    running_policies = set()
    done_policies = set()

    # Load the mapping for UIDs to logits indices
    with open(inference_args['uid_to_logits'], 'r') as f:
        uid_to_logits = json.load(f)


    buffer_usage_bar = tqdm(smoothing=0, unit='slots', total=result_buffer.size)
    buffer_usage_bar.set_description('Buffer left')
    rendering_bar = tqdm(smoothing=0.1, unit=' images')
    rendering_bar.set_description('Rendering')
    policies_bar = tqdm(total=len(policy_controllers), unit=' policies')
    policies_bar.set_description('Policies')
    work_queue = {}
    renders_to_report = 0
    last_tqdm = 0

    wait_before_start_new = False


    while True:
        if len(done_policies) == len(policy_controllers):
            break  # We finished all the policies

        # Pulling all the work to be done
        pulled_count = 0
        for policy in running_policies:
            pulled = policy.pull_work()
            if pulled is None:
                continue
            pulled_count += 1
            if pulled_count > 10:  # Do not pull too much work at once, it stalls the main thread
                break
            wait_before_start_new = False
            work_queue[pulled.id] = (policy, pulled, 0, time.time())

        # If there is not enough work we start a new policy
        if len(work_queue) < 2 * len(seen_workers) and not wait_before_start_new and len(controllers_to_start):
            selected_policy = controllers_to_start.pop()
            selected_policy.start()
            running_policies.add(selected_policy)
            wait_before_start_new = True

        message = my_recv(socket, result_buffer)

        wid = message['worker_id']
        seen_workers.add(wid)

        if message['kind'] == 'info':
            socket.send_pyobj({
                'kind': 'info',
                'environments': list_envs,
                'models': list_models,
                'render_args': render_args,
                'uid_to_logits': uid_to_logits,
                'inference': inference_args,
                'controls_args': controls_args
            })
        elif message['kind'] == 'pull':
            bs = message['batch_size']
            last_env = message['last_environment']
            last_model = message['last_model']

            to_work_on = []
            to_send = []

            def custom_order(arg):
                policy, job, num_scheduled, time_scheduled = arg
                return (num_scheduled,
                        (job.environment != last_env) + (job.model != last_model),
                        time_scheduled, job.id)


            to_work_on = sorted(work_queue.values(), key=custom_order)[: bs]

            for _, job,  __, ___ in to_work_on:
                policy, job, num_scheduled, time_scheduled = work_queue[job.id]
                to_send.append(job)
                # Remember that we sent this job to one extra worker
                work_queue[job.id] = (policy, job, num_scheduled + 1, time_scheduled)

            # Send the job information to the worker node
            socket.send_pyobj({
                'kind': 'work',
                'params_to_render': to_send,
            })

        elif message['kind'] == 'push':
            # Extract the result from the message
            jobid, result = message['job'], message['result']

            if jobid in work_queue:
                # Recover the policy associated to this job entry
                selected_policy, job, _, _ = work_queue[jobid]
                del work_queue[job.id]  # This is done do not give it to anyone else 
                selected_policy.push_result(job.id, result)
            else:
                pass  # This task has been completed earlier by another worker

            socket.send_pyobj({
                'kind': 'ack'
            })

            renders_to_report += 1

            if time.time() > last_tqdm + TQDM_FREQ:
                last_tqdm = time.time()
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
                    'pending': len(work_queue)
                })

        else:
            socket.send_pyobj({
                'kind': 'bad_query'
            })

        for policy in list(running_policies):
            if not policy.is_alive():
                policies_bar.update(1)
                running_policies.remove(policy)
                done_policies.add(policy)

    rendering_bar.close()
    policies_bar.close()
    buffer_usage_bar.close()

    print("==>[Received all the results]")
    print("==>[Shutting down workers]")
    for _ in tqdm(range(len(seen_workers)), desc='Shutting down', unit=' workers'):
        message = my_recv(socket, result_buffer)
        socket.send_pyobj({
            'kind': 'die'
        })
        try:
            wid = message['worker_id']
            del seen_workers[wid]
        except:  # This worker didn't do work yet, we still shut it down
            pass
