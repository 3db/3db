from torchvision.ops import boxes
import torch as ch

class SimpleDetectionEvaluator:
    def __init__(self, iou_threshold, min_recall=1.0, min_precision=0.0):
        self.iou_threshold = iou_threshold
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.label_type = "segmentation_map"

    def get_bounding_boxes(self, seg_map):
        BG_IND = 0
        assert seg_map.dtype == 'uint16', 'Wrong dtype for segmentation map'
        unique_objs = seg_map.unique()
        assert BG_IND in unique_objs, 'Background class (-1) not found'
        bbs = []
        for obj in unique_objs:
            if obj == BG_IND: continue 
            filt = seg_map == obj
            rows, cols = [np.any(filt, axis=a) for a in (0, 1)]
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Classes are one-indexed
            bbs.append([rmin, cmin, rmax, cmax, obj - 1])
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
        all_ious = boxes.box_iou(pred['boxes'], gt_boxes[:,4]) 
        hits = all_ious > self.iou_threshold
        # TODO: check that i got precision and recall right
        prec, recall = [ch.any(hits, dim=d).mean().item() for d in (0, 1)]
        return (prec >= self.min_precision) and (recall >= self.min_recall)

  
    def loss(self, pred, lab):
        """
        Arguments:
        - pred (dict) : same format as default torchvision models, must have 
            keys ('boxes', 'labels', 'scores'), and 
        - lab (np.array) : segmentation map containing the ground-truth objects
        """
        # TODO: do this
               
