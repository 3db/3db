"""
threedb.evaluators.classification
==================================

Implements a basic evaluator for classification-based tasks.
"""

import json
from typing import Dict, List, Tuple, Any

import torch as ch
from torch import nn, Tensor
from typeguard import check_type

from threedb.evaluators.base_evaluator import BaseEvaluator, Output, LabelType

class SimpleClassificationEvaluator(BaseEvaluator):
    """SimpleClassificationEvaluator

    A concrete implementation of the abstract
    :class:`threedb.evaluators.base_evaluator.BaseEvaluator` that is designed
    for classification tasks.
    """
    output_type = 'classes'
    KEYS = ['is_correct', 'loss', 'prediction']

    def __init__(self, *, topk: int, classmap_path: str):
        """Initialize an Evaluator for classification

        Parameters
        ----------
        topk : int

        classmap_path : str
            a path to a JSON file mapping model UIDs to class numbers.
        """
        super().__init__()
        self.crit: nn.Module = nn.CrossEntropyLoss()
        self.topk: int = topk
        self.uid_to_targets: Dict[str, LabelType] = json.load(open(classmap_path))
        check_type('uid_to_targets', self.uid_to_targets, Dict[str, LabelType])

    def get_segmentation_label(self, model_uid: str) -> int:
        label = self.uid_to_targets[model_uid]
        if isinstance(label, list):
            return label[0]
        return label

    def get_target(self, model_uid: str, render_output: Dict[str, Tensor]) -> LabelType:
        """See the docstring of
        :meth:`threedb.evaluators.base_evaluator.BaseEvaluator.get_target`.
        """
        return self.uid_to_targets[model_uid]

    def declare_outputs(self) -> Dict[str, Tuple[List[int], str]]:
        """See the docstring of
        :meth:`threedb.evaluators.base_evaluator.BaseEvaluator.declare_outputs`.

        Returns
        -------
        Dict[str, Tuple[List[int], type]]

        """
        return {
            'is_correct': ([], 'bool'),
            'loss': ([], 'float32'),
            'prediction': ([self.topk], 'int32')
        }

    def summary_stats(self, pred: ch.Tensor, label: LabelType) -> Dict[str, Output]:
        """Concrete implementation of
        :meth:`threedb.evaluators.base_evaluator.BaseEvaluator.summary_stats`
        (see that docstring for information on the abstract function). Returns
        correctness (binary value) and cross-entropy loss of the prediction.

        Parameters
        ----------
        pred : ch.Tensor
            The output of the inference model: expected to be a 1D tensor of
            size (n_classes).
        label : LabelType
            An integer or list of integers representing the target label

        Returns
        -------
        Dict[str, Output]
            A dictionary containing the results to log from this evaluator,
            namely the correctness and the cross-entropy loss.
        """
        if isinstance(label, int):
            label = [label]
        _, topk_inds = pred.topk(self.topk)
        is_correct = any([pred_lab in label for pred_lab in topk_inds])
        stacked_pred: Tensor = ch.stack([pred for _ in range(len(label))])
        im_loss: float = float(self.crit(stacked_pred, ch.tensor(label)).item())
        return {
            'is_correct': is_correct,
            'loss': im_loss,
            'prediction': topk_inds
        }

    def to_tensor(self, pred: Any, *_) -> Tensor:
        return pred

Evaluator = SimpleClassificationEvaluator
