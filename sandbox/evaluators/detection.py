from torchvision.ops import boxes
from sandbox.evaluators import base_evaluator
import torch as ch

from PIL import Image 


class SimpleDetectionEvaluator(base_evaluator.Evaluator):
    label_type = 'segmentation_map'
    output_type = 'bboxes'

    def __init__(self, iou_threshold, min_recall=1.0, min_precision=0.0):
        self.iou_threshold = iou_threshold
        self.min_recall = min_recall
        self.min_precision = min_precision

    def get_bounding_boxes(self, seg_map):
        BG_IND = -1
        unique_objs = seg_map.unique()
        assert BG_IND in unique_objs, f'Background class ({BG_IND}) not found (found {unique_objs})'
        bbs = []
        for obj in unique_objs:
            # TODO: there's a more efficient way to do the below 
            if obj == BG_IND: continue 
            filt = seg_map == obj
            _, cols, rows = ch.where(filt)
            bbs.append([rows.min(), cols.min(), rows.max(), cols.max(), obj])
        return ch.tensor(bbs)
        
    def is_correct(self, pred, lab):
        """
        Arguments:
        - pred (dict) : same format as default torchvision models, must have 
            keys ('boxes', 'labels', 'scores'), and 
        - lab (np.array) : segmentation map containing the ground-truth objects

        Returns true if the objects in the ground-truth have been captured
        with precision (TP / TP + FP) at least equal to ``min_precision`` and
        recall (TP / TP + FN) at least ``min_recall``.
        """
        gt_boxes = self.get_bounding_boxes(lab)
        keep_inds = boxes.nms(pred['boxes'], pred['scores'], self.iou_threshold)
        all_ious = boxes.box_iou(pred['boxes'][keep_inds], gt_boxes[:,:4]) 
        hits = all_ious > self.iou_threshold
        # TODO: check that i got precision and recall right
        recall, precision = [ch.any(hits, dim=d).float().mean().item() for d in (0, 1)]
        print(f"Precision: {precision} | Recall: {recall}")
        return (precision >= self.min_precision) and (recall >= self.min_recall)

  
    def loss(self, pred, lab):
        """
        Arguments:
        - pred (dict) : same format as default torchvision models, must have 
            keys ('boxes', 'labels', 'scores'), and 
        - lab (np.array) : segmentation map containing the ground-truth objects
        """
        # TODO: do this
    
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