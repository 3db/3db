import threading
from queue import Queue, Empty
from collections import namedtuple
from uuid import uuid4
import numpy as np

from sandbox.utils import init_policy, obtain_prediction


JobDescriptor = namedtuple("JobDescriptor", ['order', 'id', 'environment',
                                             'model', 'render_args',
                                             'control_order'])

class RemoteEvaluator(threading.Thread):

    def __init__(self, index, result_queue, remote_server_url, image):
        super().__init__()
        self.index = index
        self.image = image
        self.result_queue = result_queue
        self.remote_server_url = remote_server_url

    def run(self):
        result = obtain_prediction(self.remote_server_url, self.image)
        self.result_queue.put((self.index, result), block=True)



class PolicyController(threading.Thread):

    def __init__(self, env_file, search_space, model_name, policy_args,
                 inference_server, max_batch_size=100):
        super().__init__()
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.env_file = env_file
        self.model_name = model_name
        self.policy_args = policy_args
        self.search_space = search_space
        self.inference_server = inference_server

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

            result = [None] * len(args)
            predictions = [None] * len(args)
            prediction_queue = Queue()

            evaluator_threads = [None] * len(args)

            # Waiting and reordering the results
            for _ in range(len(args)):
                descriptor, job_result = self.result_queue.get(block=True)
                result[descriptor.order] = job_result
                thread = RemoteEvaluator(descriptor.order,
                                         prediction_queue,
                                         self.inference_server,
                                         job_result)
                thread.start()
                evaluator_threads.append(thread)

            for _ in range(len(args)):
                order, prediction = prediction_queue.get(block=True)
                predictions[order] = prediction

            result = np.stack(result), np.array(predictions)
            return result

        policy = init_policy(self.policy_args)
        policy.run(render)
