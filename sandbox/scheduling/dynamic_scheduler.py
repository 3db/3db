import zmq
import itertools
import os
import time
import numpy as np
import random
from tqdm import tqdm
import torch as ch
import json
from sandbox.scheduling import policy_controller
from sandbox.utils import BigChungusCyclicBuffer
from typing import List, Dict, Any, Optional
from sandbox.scheduling.policy_controller import PolicyController
from sandbox.scheduling.search_space import SearchSpace
from sandbox.log import Logger, LoggerManager

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

def my_recv(socket: zmq.Socket, 
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

def schedule_work(port: int, 
                  max_running_policies: int, 
                  environments: List[str], 
                  list_models: List[str], 
                  controls: List[Any],
                  config: Dict[str, Dict[str, Any]],
                  result_buffer: BigChungusCyclicBuffer,
                  logger_manager: LoggerManager,
                  single_model: bool):
                #   render_args: Dict[str, Any], inference_args, controls_args,
                #   evaluation_args):
                #   result_buffer: BigChungusCyclicBuffer):

    # Open a socket for communicating with the clients
    context = zmq.Context(io_threads=1)
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    seen_workers = set()
    policy_controllers = []

    search_space = SearchSpace(controls)
    continuous_dim, discrete_sizes = search_space.generate_description()

    controllers_to_start = policy_controllers[:]
    running_policies = set()
    done_policies = set()

    # Load the mapping for UIDs to target indices
    with open(config['inference']['uid_to_targets'], 'r') as f:
        uid_to_targets = json.load(f)

    buffer_usage_bar = tqdm(smoothing=0, unit='slots', total=result_buffer.size)
    buffer_usage_bar.set_description('Buffer left')
    rendering_bar = tqdm(smoothing=0.1, unit=' images')
    rendering_bar.set_description('Rendering')
    policies_bar = tqdm(total=len(policy_controllers), unit=' policies')
    policies_bar.set_description('Policies')
    work_queue = {}
    renders_to_report, valid_renders, total_renders, last_tqdm = 0, 0, 0, 0

    wait_before_start_new = False

    info_to_send = {
        'kind': 'info',
        'environments': environments,
        'models': list_models,
        'render_args': config['render_args'],
        'uid_to_targets': uid_to_targets,
        'inference': config['inference'],
        'controls_args': config['controls'],
        'evaluation_args': config['evaluation']
    }

    while True:
        if len(done_policies) == len(policy_controllers) > 0:
            break  # We finished all the policies

        message = my_recv(socket, result_buffer)
        wid = message['worker_id']
        seen_workers.add(wid)

        if result_buffer.initialized:

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
            if len(work_queue) < 2 * len(seen_workers) and not wait_before_start_new and len(controllers_to_start) and len(running_policies) < max_running_policies:
                selected_policy = controllers_to_start.pop()
                selected_policy.start()
                running_policies.add(selected_policy)
                wait_before_start_new = True
        else:
            assert message['kind'] in {'info', 'decl'}, \
                'message #1 was not "kind" == "info" or "decl", maybe race condition?'

        if message['kind'] == 'info':
            socket.send_pyobj(info_to_send)
        elif message['kind'] == 'decl':
            matches_decl = result_buffer.declare_buffers(message['declared_outputs'])
            assert matches_decl, 'Attempted to initialize buffer with different channels'
            socket.send_pyobj({'kind': 'ack'})

            for env, model in tqdm(list(itertools.product(environments, list_models)), desc="Init policies"):
                env = env.split('/')[-1]
                model = model.split('/')[-1]
                policy_controllers.append(
                    PolicyController(env, search_space, model, {
                        'continuous_dim': continuous_dim,
                        'discrete_sizes': discrete_sizes,
                        **config['policy']}, logger_manager, result_buffer))
                if single_model: 
                    break
            
            controllers_to_start = policy_controllers[:]

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

            total_renders += 1

            if jobid in work_queue:
                # Recover the policy associated to this job entry
                selected_policy, job, _, _ = work_queue[jobid]
                del work_queue[job.id]  # This is done do not give it to anyone else 
                selected_policy.push_result(job.id, result)
                renders_to_report += 1
                valid_renders += 1
            else:
                # This task has been completed earlier by another worker
                result_buffer.free(result, -1) # We have to free the result

            socket.send_pyobj({
                'kind': 'ack'
            })

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
                    'pending': len(work_queue),
                    'waste%': (1 - valid_renders / total_renders) * 100
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
