import importlib

def init_module(description):
    args = {k: v for (k, v) in description.items() if k != 'module'}
    module = importlib.import_module(description['module'])
    try:
        return module.Control(**args)
    except AttributeError:
        return module.Policy(**args)

