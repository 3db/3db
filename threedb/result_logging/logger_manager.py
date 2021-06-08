"""
threedb.result_logging.logger_manager
=====================================

"""

from threedb.result_logging.base_logger import BaseLogger

class LoggerManager():
    """
    A LoggerManager allows us to log from several loggers at once, without
    handling each one individually.

    Users should not have to modify or subclass this to extend 3DB.
    """
    def __init__(self):
        super().__init__()
        self.loggers = []

    def append(self, logger: BaseLogger):
        """
        Adds a new logger.
        """
        self.loggers.append(logger)

    def log(self, item):
        """
        Logs the given items from each logger under management.
        """
        for logger in self.loggers:
            logger.enqueue(item)

    def start(self):
        """
        Starts each logger under management.
        """
        for logger in self.loggers:
            logger.start()

    def join(self):
        """
        Wait for each logger under management to complete their tasks.
        """
        for logger in self.loggers:
            logger.join()