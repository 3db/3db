"""
threedb.result_logging.tb_logger
==================================

Subclass of :mod:`threedb.result_logging.base_logger.BaseLogger`.
"""

import pandas as pd

from threedb.result_logging.base_logger import BaseLogger
from threedb.utils import CyclicBuffer
from torch.utils.tensorboard import SummaryWriter

class TbLogger(BaseLogger):
    """
    TensorBoard Logger.

    Logs accuracy (as a scalar) and all generated images to TensorBoard.
    """
    def __init__(self, tb_dir: str, result_buffer: CyclicBuffer, _) -> None:
        super().__init__(tb_dir, result_buffer, None)
        self.tb_dir = tb_dir
        print(f'==> [Logging tensorboard to {tb_dir}]')
        self.writer = None # Defer allocation in the sub-process
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
        self.numeric_data = []
        self.images = {}
        self.count = 0

    def _write(self):
        """
        Internal function for writing to the tensorboard.
        """
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)
        data_df = pd.DataFrame(self.numeric_data)
        current_acc = data_df.is_correct.mean()
        self.writer.add_scalar('Accuracy', current_acc, self.count)
        for uid in data_df.model.unique():
            image_id = data_df[data_df.model == uid].id.iloc[-1]
            self.writer.add_image(uid, self.images[image_id], self.count)
        self.images = {}
        self.numeric_data = []

    def log(self, item):
        self.count += 1
        rix = item['result_ix']
        buf_data = self.result_buffer[rix]
        print(buf_data.keys(), item.keys())
        information = {k: v for k, v in item.items() if k != 'result_ix'}
        information['is_correct'] = buf_data

        self.numeric_data.append(information)
        self.images[item['id']] = image['rgb'].clone()

        if self.count % 1 == 0:
            self._write()

        self.result_buffer.free(rix, self.regid)
    
    def end(self) -> None:
        pass

Logger = TbLogger