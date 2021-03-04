from threedb.result_logging.base_logger import BaseLogger
from os import path
import cv2
import numpy as np

class ImageLogger(BaseLogger):
    """
    This logger [TODO]
    """
    def __init__(self, save_dir, result_buffer, _):
        super().__init__(save_dir, result_buffer, None)
        self.result_buffer = result_buffer
        self.regid = self.result_buffer.register()
        self.folder = save_dir
        print(f'==> [Logging images to {dir} with regid {self.regid}]')

    def log(self, item):
        rix = item['result_ix']
        buf_data = self.result_buffer[rix]
        for channel_name in ['rgb']: 
            image = buf_data[channel_name]
            if channel_name == 'segmentation':
                img_path = path.join(self.folder,
                                     item['id'] + '_' + channel_name + '.npy')
                np.save(img_path, image.numpy()[0])
            else:
                img_path = path.join(self.folder,
                                     item['id'] + '_' + channel_name + '.png')
                img_to_write = cv2.cvtColor(image.permute(1,2,0).numpy()*255.0, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, img_to_write)
        self.result_buffer.free(rix, self.regid)

Logger = ImageLogger