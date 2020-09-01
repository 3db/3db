import threading
from queue import Queue, Empty
from collections import namedtuple

from uuid import uuid4
from utils import init_module



JobDescriptor = namedtuple("JobDescriptor", ['order', 'id', 'environment',
                                             'model', 'render_args'])


class PolicyController(threading.Thread):

    def __init__(self, env_file, search_space, model_name, policy_args, max_batch_size=100):
        super().__init__()
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.env_file = env_file
        self.model_name = model_name
        self.policy_args = policy_args
        self.search_space = search_space

    def pull_work(self, worker_id):
        print("pulling")
        try:
            print((self.work_queue.qsize()))
            return self.work_queue.get(block=False)
        except Empty:
            return None

    def push_result(self, descriptor, result):
        self.result_queue.put((descriptor, result))

    def run(self):
        def render(args):
            # Posting the jobs to the queue
            for i, (continuous_args, discrete_args) in enumerate(args):

                argument_dict = self.search_space.unpack(continuous_args,
                                                         discrete_args)
                descriptor = JobDescriptor(order=i, id=str(uuid4()),
                                           render_args=argument_dict,
                                           environment=self.env_file,
                                           model=self.model_name)
                self.work_queue.put(descriptor, block=True)
            print("everything is in the queue")

            result = [None] * len(args)

            # Waiting and reordering the results
            for _ in range(len(args)):
                descriptor, job_result = self.result_queue.get(block=True)
                result[descriptor.order] = job_result
            return result

        policy = init_module(self.policy_args)
        print("About to start")
        policy.run(render)

