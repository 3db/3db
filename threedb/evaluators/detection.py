"""
threedb.evaluators.detection
============================

An evaluator for object detection models.

Includes a concrete implementation of
:class:`threedb.evaluators.base_evaluator.BaseEvaluator`.
"""


import json
from typing import Any, Dict, List, Tuple

import torch as ch
from torch.tensor import Tensor
from torchvision.ops import boxes
from typeguard import check_type

from threedb.evaluators.base_evaluator import BaseEvaluator, LabelType, Output

# Default segmentation map value set by blender
BG_IND = -1

class SimpleDetectionEvaluator(BaseEvaluator):
    """Concrete implementation of
    :class:`threedb.evaluators.base_evaluator.BaseEvaluator`.
    """

    output_type = 'bboxes'
    output_shape = [100, 6]
    KEYS = ['is_correct_nolabel', 'precision_nolabel', 'recall_nolabel',
            'is_correct', 'precision', 'recall', 'is_valid']

    def __init__(self, iou_threshold: float, classmap_path: str,
                 min_recall: float = 1.0, min_precision: float = 0.0):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.min_rec = min_recall
        self.min_prec = min_precision
        self.uid_to_targets: Dict[str, LabelType] = json.load(open(classmap_path))
        check_type('uid_to_targets', self.uid_to_targets, Dict[str, LabelType])

    def get_segmentation_label(self, model_uid: str) -> int:
        label = self.uid_to_targets[model_uid]
        return label[0] if isinstance(label, list) else label

    def declare_outputs(self) -> Dict[str, Tuple[List[int], str]]:
        return {
            'is_correct': ([], 'bool'),
            'precision': ([], 'float32'),
            'recall': ([], 'float32'),
            'is_correct_nolabel': ([], 'bool'),
            'precision_nolabel': ([], 'float32'),
            'recall_nolabel': ([], 'float32'),
            'is_valid': ([], 'bool')
        }

    def get_target(self,
                   model_uid: str,
                   render_output: Dict[str, Tensor]) -> ch.Tensor:
        """A label generator for object detection, using the outputted
        segmentation map. Uses a simplistic approach that assumes that each
        label in the segmentation map is a unique object. Specifically, for
        each unique label (object type) in the segmentation map, we make one
        bounding box, defined as the minimum-size box containing all pixels
        of this type in the segmentation map.

        Parameters
        ----------
        model_uid : str
            The UID of the model (not used)
        render_output : Dict[str, Tensor]
            The output of the renderer
            (:meth:`threedb.rendering.base_renderer.BaseRenderer.render_and_apply`)
            for this model. Must have a key called 'segmentation' containing
            the object segmentation.

        Returns
        -------
        ch.Tensor
            A set of bounding boxes and labels for the objects that should be
            detected. In particular, the bounding boxes are stored in ``(x1, y1,
            x2, y2, label)`` format.
        """
        seg_map = render_output['segmentation']
        unique_objs = list(map(int, seg_map.unique()))
        assert BG_IND in unique_objs, f'Background class ({BG_IND}) not found (found {unique_objs})'
        bbs = []
        for obj in unique_objs:
            if obj == BG_IND:
                continue
            filt = seg_map == obj
            _, cols, rows = ch.where(filt)
            bbs.append([rows.min(), cols.min(), rows.max(), cols.max(), obj])
        return ch.tensor(bbs)

    def summary_stats(self,
                      pred: Dict[str, ch.Tensor],
                      label: ch.Tensor) -> Dict[str, Output]:
        """Concrete implementation of
        :meth:`threedb.evaluators.base_evaluator.BaseEvaluator.summary_stats`

        Parameters
        ----------
        pred : Dict[str, ch.Tensor]
            Same output format as default torchvision detection models in
            evaluation mode, must have keys ('boxes', 'labels', 'scores')
        label : ch.Tensor
            Segmentation map containing the ground-truth objects

        Returns
        -------
        Dict[str, Output]
            The model's performance on the given image. Non-maximal suppression
            is performed on the output of the classifier, the precision and
            recall are calculated using the IOU threshold set at
            instantiation-time (see `here <https://en.wikipedia.org/wiki/Precision_and_recall>`_ 
            for information on precision and recall in object detection).

            Precision and recall are then thresholded (by the
            ``min_precision`` and ``min_recall`` parameters) to get a single
            boolean representing correctness. We return the corresponding
            keys ``precision_nolabel``, ``recall_nolabel``, and
            ``is_correct_nolabel`` as well as their counterparts
            ``precision``, ``recall`` and ``is_correct`` which take both box
            positions and class labels into account (the former only evaluate
            localization, not labelling).

            Finally, we return a key ``is_valid`` that represents whether we
            the label corresponding to the image is actually a valid class
            label.
        """
        keep_inds = boxes.nms(pred['boxes'], pred['scores'], self.iou_threshold)
        all_ious = boxes.box_iou(pred['boxes'][keep_inds], label[:, :4])
        iou_hits = (all_ious > self.iou_threshold)
        label_hits = label[:, 4][None].eq(pred['labels'][keep_inds][:, None])
        assert label_hits.shape == iou_hits.shape
        hits = iou_hits & label_hits
        rec_nl, prec_nl = [float(ch.any(iou_hits, dim=d).float().mean().item()) for d in (0, 1)]
        rec, prec = [float(ch.any(hits, dim=d).float().mean().item()) for d in (0, 1)]
        return {
            'is_correct_nolabel': (prec_nl >= self.min_prec) and (rec_nl >= self.min_rec),
            'precision_nolabel':  prec_nl,
            'recall_nolabel': rec_nl,
            'is_correct': (prec >= self.min_prec) and (rec >= self.min_rec),
            'precision': prec,
            'recall': rec,
            'is_valid': (int(label[:, 4]) != -1)
        }

    def to_tensor(self, pred: Any, output_shape: List[int], input_shape: List[int]) -> Tensor:
        """Concrete implementation of
        :meth:`threedb.evaluators.base_evaluator.BaseEvaluator.to_tensor`.

        Turns a prediction dictionary into a tensor with the given output_shape
        (N x 6). To do this, we concatenate the prediction into the form ``[(x1,
        y1, x2, y2, score, label)]``.
        """
        _, height, width = input_shape
        out = ch.zeros(*output_shape) - 1
        keep_inds = boxes.nms(pred['boxes'], pred['scores'], self.iou_threshold)
        num_boxes = keep_inds.shape[0]
        if num_boxes == 0:
            return out
        keys = ('boxes', 'scores', 'labels')
        kept_preds = [pred[s][keep_inds].view(num_boxes, -1).float() for s in keys]
        out[:num_boxes] = ch.cat(kept_preds, dim=1)
        out[:, [0, 2]] /= width
        out[:, [1, 3]] /= height
        return out

Evaluator = SimpleDetectionEvaluator
