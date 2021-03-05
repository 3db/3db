import torch as ch
from abc import ABC, abstractmethod

class Evaluator(ABC):
    """Abstract Base Class for Evaluators

    An evaluator is in charge of taking the output of an inference model,
    analyzing it, and returning the relevant summary outputs (e.g.,
    correctness for classification, IoU for detection, MSE for segmentation,
    etc.)
    """
    def __init__(self):
        pass

    @abstractmethod
    def is_correct(self, pred, label):
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, label):
        raise NotImplementedError

    @abstractmethod
    def extra_info(self, pred, label):
        raise NotImplementedError

    def to_tensor(self, pred, output_shape, input_shape): 
        if isinstance(pred, ch.Tensor):
            assert list(pred.shape) == list(output_shape), \
                f"Shape of prediction ({pred.shape}) does not match declared shape ({output_shape})"
            return pred
        ERR_MSG = "Must implement evaluator.to_tensor if prediction is not already a tensor"
        raise ValueError(ERR_MSG)
