from torch import nn
import torch as ch
from sandbox.evaluators import base_evaluator

class SimpleClassificationEvaluator(base_evaluator.Evaluator): 
    def __init__(self, *, topk):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()
        self.topk = topk
        self.label_type = "classes"

    def fix_type(self, label):
        """
        Takes a label in any valid format (list of ints/tensors, single int, single tensor).
        Returns a unified format (list of integers).
        """
        assert isinstance(label, (int, list, ch.Tensor)), \
            "Label must be an int, list, or torch tensor"
        if isinstance(label, int):
            return [label]
        elif isinstance(label, ch.Tensor):
            if len(label.shape) == 0: return [label.item()]
            else: return label.tolist()
        return label

    def is_correct(self, pred, label):
        """
        Takes in a prediction (tensor of size (# logits)) and a label (list, int, or tensor).
        Returns true if any of the valid labels is within the topk predicted labels.
        """
        label = self.fix_type(label)
        return any([pred_lab in label for pred_lab in pred.topk(self.topk).indices])
    
    def loss(self, pred, label):
        """
        Takes in a prediction (tensor of size (# logits)) and a label (list, int, or tensor).
        Returns the minimum (cross-entropy) loss over valid target labels.
        """
        label = self.fix_type(label)
        return min([self.crit(pred[None,...], ch.tensor([l])) for l in label])

Evaluator = SimpleClassificationEvaluator