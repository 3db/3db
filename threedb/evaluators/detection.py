from typing import Dict, Tuple, List
from torch.tensor import Tensor
from torchvision.ops import boxes
from threedb.evaluators.base_evaluator import BaseEvaluator, Output, LabelType
from typeguard import check_type
import torch as ch
import json

from PIL import Image 

# Default segmentation map value set by blender
BG_IND = -1

class SimpleDetectionEvaluator(BaseEvaluator):
    output_type = 'bboxes'
    output_shape = [100, 6]
    KEYS = ['is_correct_nolabel', 'precision_nolabel', 'recall_nolabel',
            'is_correct', 'precision', 'recall', 'is_valid']

    def __init__(self, iou_threshold: float, classmap_path: str,
                 min_recall: float=1.0, min_precision: float=0.0):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.min_recall = min_recall
        self.min_precision = min_precision
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
            if obj == BG_IND: continue 
            filt = seg_map == obj
            _, cols, rows = ch.where(filt)
            bbs.append([rows.min(), cols.min(), rows.max(), cols.max(), obj])
        return ch.tensor(bbs)

    def summary_stats(self,
                      pred: Dict[str, ch.Tensor], 
                      label: ch.Tensor) -> Dict[str, Output]:
        """
        Arguments:
        - pred (dict) : same format as default torchvision models, must have 
            keys ('boxes', 'labels', 'scores'), and 
        - lab (np.array) : segmentation map containing the ground-truth objects

        Returns true if the objects in the ground-truth have been captured
        with precision (TP / TP + FP) at least equal to ``min_precision`` and
        recall (TP / TP + FN) at least ``min_recall``.
        """
        keep_inds = boxes.nms(pred['boxes'], pred['scores'], self.iou_threshold)
        all_ious = boxes.box_iou(pred['boxes'][keep_inds], label[:,:4]) 
        iou_hits = (all_ious > self.iou_threshold)
        label_hits = label[:,4][None].eq(pred['labels'][keep_inds][:, None])
        assert label_hits.shape == iou_hits.shape
        hits = iou_hits & label_hits
        recall, precision = [float(ch.any(hits, dim=d).float().mean().item()) for d in (0, 1)]
        print(f"Precision: {precision} | Recall: {recall}")
        return {
            'is_correct_nolabel': (precision >= self.min_precision) and (recall >= self.min_recall),
            'precision_nolabel':  precision,
            'recall_nolabel': recall
        }

    def to_tensor(self, pred, output_shape, input_shape):
        """
        Turns a prediction dictionary into a tensor with the given output_shape (N x 6).
        To do this, we concatenate the prediction into the form [(x1, y1, x2, y2, score, label)].
        """
        C, H, W = input_shape
        out = ch.zeros(*output_shape) - 1
        keep_inds = boxes.nms(pred['boxes'], pred['scores'], self.iou_threshold)
        N = keep_inds.shape[0]
        keys = ('boxes', 'scores', 'labels')
        out[:N] = ch.cat([pred[s][keep_inds].view(N, -1).float() for s in keys], dim=1)
        out[:,[0, 2]] /= W
        out[:,[1, 3]] /= H
        return out 
               
Evaluator = SimpleDetectionEvaluator