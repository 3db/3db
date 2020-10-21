
class BaseControl:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def apply(self):
        raise NotImplementedError