from multiprocessing import Process, Queue
from typing import Any, Dict, Optional
from sandbox.utils import BigChungusCyclicBuffer
from abc import abstractmethod, ABC

class BaseLogger(Process, ABC):
    """
    Abstract class for a Logger, inherits from ``multiprocessing.Process``.
    Implements the additional functions:

    - enqueue(): put a new item on the logging queue, to be logged ASAP
    - log(): the actual logging mechanism, meant to be overwritten by the
        user for each unique subclass. This should not be called directly,
        but rather will be called by ``run()``
    - run(): main loop, waits for logs to be added to the queue, and calls
        ``log()`` on them.
    [TODO]
    """
    def __init__(self, root_dir: str, 
                       result_buffer: BigChungusCyclicBuffer, 
                       config: Optional[Dict[str, Dict[str, Any]]]) -> None:
        super().__init__()
        self.root_dir: str = root_dir
        self.buffer = result_buffer
        self.config: Optional[Dict[str, Dict[str, Any]]] = config
        self.queue: Queue = Queue()

    def enqueue(self, item: dict) -> None:
        """
        Add an item to the queue to be logged. See the documentation of
            :meth:``[TODO]`` for the required structure of the log item.
        """
        self.queue.put(item)

    @abstractmethod
    def log(self, item: dict) -> None:
        """
        Log an item.

        Arguments:
        - item (dict) : must have keys: ``outputs``, ``is_correct`` and ``output_type``:
            - ``outputs`` should be a tensor of model outputs (predictions)
            - ``outputs TODO
        """
        raise NotImplementedError

    def end(self) -> None:
        """
        Performs cleanup operations for the logger. No-op by default, should be
        overriden with code for closing any open file handles, ports, etc.
        """
        pass

    def run(self) -> None:
        assert self.buffer.initialized, \
            'Tried to start logger before buffer was initialized'
        while True:
            item = self.queue.get()
            if item is None:
                break
            self.log(item)
        self.end()