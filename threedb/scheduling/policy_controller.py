"""
threedb.scheduling.policy_controller
====================================
"""

from multiprocessing import Process, Queue
from queue import Empty
from collections import namedtuple
from uuid import uuid4
import numpy as np
from typing import List, Dict, Optional, Any
from threedb.scheduling.search_space import SearchSpace
from threedb.result_logging.logger_manager import LoggerManager
from threedb.utils import init_policy, CyclicBuffer


JobDescriptor = namedtuple("JobDescriptor", ['order', 'id', 'environment',
                                             'model', 'render_args',
                                             'control_order'])

class PolicyController(Process):

    def __init__(self, search_space: SearchSpace,
                       env_file: str,
                       model_name: str,
                       policy_args: Dict[str, Any],
                       logger_manager: LoggerManager,
                       result_buffer: CyclicBuffer):
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

            client_results: List[Optional[dict]] = [None] * len(args)

            # Waiting and reordering the results
            for _ in range(len(args)):
                job_id, result_ix = self.result_queue.get(block=True)
                c_result = self.result_buffer[result_ix]
                descriptor = all_descriptors[job_id]

                self.logger_manager.log({
                    **descriptor._asdict(),
                    'result_ix': result_ix
                })
                client_results[descriptor.order] = {k: v.clone() for (k, v) in c_result.items()}
                self.result_buffer.free(result_ix, 1)

            result_keys = client_results[0].keys()
            stacked_results = {k: np.stack([res[k] for res in client_results]) for k in result_keys}
            return stacked_results

        policy = init_policy(self.policy_args)
        policy.run(render)
