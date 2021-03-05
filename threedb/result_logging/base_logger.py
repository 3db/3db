"""base_logger.py

Implements an abstract class for logging results.
"""

from multiprocessing import Process, Queue
from typing import Any, Dict, Optional
from threedb.utils import BigChungusCyclicBuffer
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
    def __init__(self,
                 root_dir: str,
                 result_buffer: BigChungusCyclicBuffer,
                 config: Optional[Dict[str, Dict[str, Any]]]) -> None:
        """Creates an instance of the logger

        Args:
            root_dir (str): where to write the logging results
            result_buffer (BigChungusCyclicBuffer): buffer where the actual
                results are written
            config (Optional[Dict[str, Dict[str, Any]]]): the 3DB experiment
                config (see TODO [link to docs])
        """
        super().__init__()
        self.root_dir: str = root_dir
        self.buffer = result_buffer
        self.config: Optional[Dict[str, Dict[str, Any]]] = config
        self.queue: Queue = Queue()

    def enqueue(self, item: Dict[str, Any]) -> None:
        """Add an item in the queue to be logged. See `here`:TODO: for
        documentation on how the log item is structured.

        Parameters
        ----------
        item : Dict[str, Any]
            The item to be logged.
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
        """Performs cleanup operations for the logger. No-op by default, should
        be overriden with code for closing any open file handles, ports, etc.
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
