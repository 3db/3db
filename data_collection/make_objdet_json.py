from argparse import ArgumentParser

COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

IMAGENET_TO_COCO = {
    'motorcycle': 670,
    'bicycle': 671,
    'bus': (654, 779, 874),
    'train'
}

IMAGENET_TO_COCO = {
    670: COCO_LABELS.index('motorcycle'), # motor scooter
    671: COCO_LABELS.index('bicycle'), # mountain bike
    654: COCO_LABELS.index('bus'), # minibus
    779: COCO_LABELS.index('bus'), # school bus
    874: COCO_LABELS.index('bus'), # trolleybus
    466: COCO_LABELS.index('train'), # bullet train

}

parser = ArgumentParser()
parser.add_argument('-i', '--input-file')