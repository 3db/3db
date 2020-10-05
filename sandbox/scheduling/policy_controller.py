import threading
from queue import Queue, Empty
from collections import namedtuple
from uuid import uuid4
import numpy as np

from sandbox.utils import init_policy


JobDescriptor = namedtuple("JobDescriptor", ['order', 'id', 'environment',
                                             'model', 'render_args',
                                             'control_order'])


class PolicyController(threading.Thread):

    def __init__(self, env_file, search_space, model_name, policy_args,
                 max_batch_size=100):
        super().__init__()
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.env_file = env_file
        self.model_name = model_name
        self.policy_args = policy_args
        self.search_space = search_space

    def pull_work(self, worker_id):
        try:
            return self.work_queue.get(block=False)
        except Empty:
            return None

    def push_result(self, descriptor, result):
        self.result_queue.put((descriptor, result))

    def run(self):
        def render(args):
            # Posting the jobs to the queue
            for i, (continuous_args, discrete_args) in enumerate(args):

                argument_dict, ctrl_list = self.search_space.unpack(continuous_args,
                                                                    discrete_args)
                descriptor = JobDescriptor(order=i, id=str(uuid4()),
                                           render_args=argument_dict,
                                           control_order=ctrl_list,
                                           environment=self.env_file,
                                           model=self.model_name)
                self.work_queue.put(descriptor, block=True)

            images = [None] * len(args)
            logits = [None] * len(args)
            is_correct = [None] * len(args)

            # Waiting and reordering the results
            for _ in range(len(args)):
                descriptor, job_result = self.result_queue.get(block=True)
                images[descriptor.order] = job_result[0]
                logits[descriptor.order] = job_result[1]
                is_correct[descriptor.order] = job_result[2]

            images = np.stack(images)
            logits = np.stack(logits)
            is_correct = np.stack(is_correct)
            return images, logits, is_correct

        policy = init_policy(self.policy_args)
        policy.run(render)
