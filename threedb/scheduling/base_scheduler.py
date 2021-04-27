"""
threedb.scheduling.base_scheduler
=================================

The task schedular of 3DB.
"""

from tqdm import tqdm
from typing import Any, Dict, Set, List
from threedb.scheduling.policy_controller import PolicyController
from threedb.scheduling.utils import recv_into_buffer
from threedb.result_logging.logger_manager import LoggerManager
from threedb.utils import CyclicBuffer

import time
import os
import json
import zmq

class Scheduler:
    def __init__(self, port: int,
                       max_running_policies: int,
                       envs: List[str],
                       models: List[str],
                       config: Dict[str, Dict[str, Any]],
                       policy_controllers: Set[PolicyController],
                       buffer: CyclicBuffer,
                       logger_manager: LoggerManager,
                       with_tqdm: bool = True) -> None:
        self.running = False

        self.envs = envs
        self.models = models
        self.buffer = buffer
        self.config = config
        self.logger_manager = logger_manager

        # Open a socket for communicating with the clients
        context = zmq.Context(io_threads=1)
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % port)

        # Keep track of the workers
        self.linked_workers: Set[str] = set()
        self.policy_controllers = policy_controllers
        self.num_policies = len(policy_controllers)
        self.done_policies = set()
        self.running_policies = set()
        self.max_running_policies = max_running_policies
        self.work_queue = {}

        # TQDM bars
        self.valid_renders, self.total_renders = 0, 0
        if with_tqdm:
            self.render_pb = tqdm(unit='images', desc='Renderings', smoothing=0.1)
            self.policies_pb = tqdm(unit='steps', desc='Policies', total=self.num_policies)

    def start(self, declared_outputs):
        """
        Inputs:
        - declared_outputs: map of key -> shape for what outputs the client will
          relay back to the server.
        
        Outputs:
        - None

        Side effects: will create the buffer using the specified shapes, start
        the policy controllers, and also the loggers.
        """
        assert self.buffer.declare_buffers(declared_outputs)
        self.socket.send_pyobj({'kind': 'ack'})
        if not self.running:
            self.logger_manager.start()
            self.running = True

    def send_info(self):
        self.socket.send_pyobj({
            'kind': 'info',
            'environments': self.envs,
            'models': self.models,
            'render_args': self.config['render_args'],
            'inference': self.config['inference'],
            'controls_args': self.config['controls'],
            'evaluation_args': self.config['evaluation']
        })

    def handle_pull(self, message: Dict[str, Any]) -> None:
        """
        Handles a "pull" request from the client, asking for work. Should send a
        new list of jobs to work on
        """
        bs = message['batch_size']
        last_env = message['last_environment']
        last_model = message['last_model']

        to_work_on = []
        to_send = []

        def custom_order(arg):
            _, job, num_scheduled, time_scheduled = arg
            return (num_scheduled,
                    int(job.environment != last_env) + int(job.model != last_model),
                    time_scheduled, job.id)

        to_work_on = sorted(self.work_queue.values(), key=custom_order)[:bs]

        for _, job,  __, ___ in to_work_on:
            policy, job, num_scheduled, time_scheduled = self.work_queue[job.id]
            to_send.append(job)
            # Remember that we sent this job to one extra worker
            self.work_queue[job.id] = (policy, job, num_scheduled + 1, time_scheduled)

        # Send the job information to the worker node
        self.socket.send_pyobj({
            'kind': 'work',
            'params_to_render': to_send,
        })

    def handle_push(self, message: Dict[str, Any]) -> None:
        # Extract the result from the message
        jobid, result = message['job'], message['result']

        self.total_renders += 1

        if jobid in self.work_queue:
            # Recover the policy associated to this job entry
            selected_policy, job, _, _ = self.work_queue[jobid]
            del self.work_queue[job.id]  # This is done do not give it to anyone else 
            selected_policy.push_result(job.id, result)
            self.render_pb.update(1)
            self.valid_renders += 1
        else:
            # This task has been completed earlier by another worker
            self.buffer.free(result, -1) # We have to free the result

        self.socket.send_pyobj({'kind': 'ack'})

    def shutdown(self):
        for _ in tqdm(range(len(self.linked_workers)), desc='Shutting down', unit=' workers'):
            message = recv_into_buffer(self.socket, self.buffer)
            self.socket.send_pyobj({
                'kind': 'die'
            })
            try:
                self.linked_workers.remove(message['worker_id'])
            except:  # This worker didn't do work yet, we still shut it down
                pass

        self.buffer.close()
        self.render_pb.close()
        self.policies_pb.close()

        # Warning the logger that we are done
        self.logger_manager.log(None)
        print("==> [Waiting for any pending logging]")

        # We have to wait until it has processed everything left in the queue
        self.logger_manager.join()
        print("==> [Have a nice day!]")

    def schedule_work(self):
        wait_before_start_new = False

        while True:
            if len(self.done_policies) == self.num_policies:
                break  # We finished all the policies

            message = recv_into_buffer(self.socket, self.buffer)
            wid = message['worker_id']
            self.linked_workers.add(wid)

            if self.running:
                pulled_count = 0
                for policy in self.running_policies:
                    pulled = policy.pull_work()
                    if pulled is None: continue
                    pulled_count += 1
                    if pulled_count > 10:  break
                    wait_before_start_new = False
                    self.work_queue[pulled.id] = (policy, pulled, 0, time.time())

                # If there is not enough work we start a new policy
                little_work = len(self.work_queue) < 2 * len(self.linked_workers)
                policies_left = len(self.policy_controllers) > 0
                running_max_policies = len(self.running_policies) >= self.max_running_policies
                if little_work and (not wait_before_start_new) and policies_left and (not running_max_policies):
                    selected_policy = self.policy_controllers.pop()
                    selected_policy.start()
                    self.running_policies.add(selected_policy)
                    wait_before_start_new = True
            else:
                assert message['kind'] in {'info', 'decl'}, \
                    'message #1 was not "kind" == "info" or "decl", maybe race condition?'

            if message['kind'] == 'info':
                self.send_info()
            elif message['kind'] == 'decl':
                self.start(message['declared_outputs'])
            elif message['kind'] == 'pull':
                self.handle_pull(message)
            elif message['kind'] == 'push':
                self.handle_push(message)
            else:
                self.socket.send_pyobj({'kind': 'bad_query'})

            for policy in list(self.running_policies):
                if not policy.is_alive():
                    self.policies_pb.update(1)
                    self.running_policies.remove(policy)
                    self.done_policies.add(policy)

            self.render_pb.set_postfix({
                'workers': len(self.linked_workers),
                'pending': len(self.work_queue),
                'waste%': (1 - self.valid_renders / max(1e-10, self.total_renders)) * 100
            })
            self.policies_pb.set_postfix({'running': len(self.running_policies)})

        print("==> [Received all the results]")
        print("==> [Shutting down workers]")
        self.shutdown()
