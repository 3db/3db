"""
Old file, will remove
"""


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

    # Load the mapping for UIDs to target indices


    work_queue = {}
    wait_before_start_new = False

        for policy in list(running_policies):
            if not policy.is_alive():
                policies_bar.update(1)
                running_policies.remove(policy)
                done_policies.add(policy)

    rendering_bar.close()
    policies_bar.close()
    buffer_usage_bar.close()