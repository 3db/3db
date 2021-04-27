"""
threedb.result_logging.base_logger
==================================

Implements an abstract class for logging results.
"""

from multiprocessing import Process, Queue
from typing import Any, Dict, Optional
from threedb.utils import CyclicBuffer
from abc import abstractmethod, ABC

class BaseLogger(Process, ABC):
    """Abstract class for a Logger, inherits from
    ``multiprocessing.Process``. Implements the additional functions:

    - ``enqueue()``: put a new item on the logging queue, to be logged ASAP
    - ``log()``: the actual logging mechanism, meant to be overwritten by the
      user for each unique subclass. This should not be called directly,
      but rather will be called by ``run()``
    - ``run()``: main loop, waits for logs to be added to the queue, and calls
      ``log()`` on them.
    - ``end()``: Performs cleanup operations for the logger. No-op by default, should
        be overriden with code for closing any open file handles, ports, etc.

    """
    def __init__(self,
                 root_dir: str,
                 result_buffer: CyclicBuffer,
                 config: Optional[Dict[str, Dict[str, Any]]]) -> None:
        """Creates a logger instance

        Parameters
        ----------
        root_dir : str
            The directory in which to write the logging results (should be the
            same for all loggers, with each logger making a subfolder to log in)
        result_buffer : CyclicBuffer
            The buffer where the main thread is writing the results
        config : Optional[Dict[str, Dict[str, Any]]]
            The config file (parsed from YAML) of the 3DB experiment being run
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
    def log(self, item: Dict[str, Any]) -> None:
        """Abstract method for logging an item from the buffer

        Parameters
        ----------
        item : Dict[str, Any]
            A dictionary containing the results of a single rendering (as
            returned by :mod:`threedb.client`): see
            [TODO] for more detailed information on what this will contiain.
        """
        raise NotImplementedError

    @abstractmethod
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
