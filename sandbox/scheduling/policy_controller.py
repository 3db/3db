from multiprocessing import Process, Queue
from queue import Empty
from collections import namedtuple
import multiprocessing
from uuid import uuid4
import cv2
import numpy as np

from sandbox.utils import init_policy


JobDescriptor = namedtuple("JobDescriptor", ['order', 'id', 'environment',
                                             'model', 'render_args',
                                             'control_order'])


class PolicyController(Process):

    def __init__(self, env_file, search_space, model_name, policy_args,
                 logger_manager, result_buffer):
        super().__init__()
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.env_file = env_file
        self.model_name = model_name
        self.policy_args = policy_args
        self.search_space = search_space
        self.logger_manager = logger_manager
        self.result_buffer = result_buffer

    def pull_work(self):
        try:
            return self.work_queue.get(block=False)
        except Empty:
            return None

    def push_result(self, descriptor, result):
        self.result_queue.put((descriptor, result))

    def run(self):
        def render(args):
            # Posting the jobs to the queue

            all_descriptors = {}
            for i, (continuous_args, discrete_args) in enumerate(args):

                argument_dict, ctrl_list = self.search_space.unpack(continuous_args,
                                                                    discrete_args)
                current_id = str(uuid4())
                descriptor = JobDescriptor(order=i, id=current_id,
                                           render_args=argument_dict,
                                           control_order=ctrl_list,
                                           environment=self.env_file,
                                           model=self.model_name)
                all_descriptors[current_id] = descriptor
                self.work_queue.put(descriptor, block=True)

            images = [None] * len(args)
            logits = [None] * len(args)
            is_correct = [None] * len(args)

            # Waiting and reordering the results
            for _ in range(len(args)):
                job_id, result_ix = self.result_queue.get(block=True)
                c_images, c_logits, c_is_correct = self.result_buffer[result_ix]
                descriptor = all_descriptors[job_id]
                self.logger_manager.log({
                    **descriptor._asdict(),
                    'result_ix': result_ix
                })
                images[descriptor.order] = {k: v.clone() for (k, v) in c_images.items()}
                logits[descriptor.order] = c_logits.clone()
                is_correct[descriptor.order] = c_is_correct
                self.result_buffer.free(result_ix)

            image_channels = list(images[0].keys())
            images = {k: np.stack([image[k] for image in images]) for k in image_channels}
            logits = np.stack(logits)
            is_correct = np.stack(is_correct)

            return images, logits, is_correct

        policy = init_policy(self.policy_args)
        policy.run(render)
