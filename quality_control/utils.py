import json
from pathlib import Path
from flatten_dict import flatten
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import torch.nn.functional as F
from robustness.tools.label_maps import CLASS_DICT
from PIL import Image
from os import path as osp
import pandas as pd 
import numpy as np

class RGBAToRGB():
    def __init__(self, fill_colour=(1., 1., 1.)):
        self.fill_colour = (int(fill_colour[0] * 255),
                            int(fill_colour[1] * 255),
                            int(fill_colour[2] * 255))

    def __call__(self, im, target=None):
        if type(im) == tuple:
            im = im[0]
        bg = Image.new('RGBA', im.size, self.fill_colour)
        return Image.alpha_composite(bg, im)

def details_df(root: Path, early_stop: int = -1):
    with open(root / 'details.log') as f:
        lines = f.readlines()
        if early_stop < 0: early_stop = len(lines)
        lines = zip(range(early_stop), lines)
        infos = [flatten(json.loads(l), reducer='path') for _, l in lines]
    infos = [x for x in infos if osp.exists(root / f"{x[ID_COL]}.png")]
    return pd.DataFrame(infos)

class RGBAToRGBWithBackground():
    def __init__(self, backgrounds=None, labels=None, 
                 mode='random', offset=None, Nclasses=10):
        assert mode in ['random', 'random_per_class', 
                        'fixed', 'fixed_per_class']

        self.backgrounds = backgrounds
        self.labels = labels
        self.mode = mode

        if mode in ['random', 'random_per_class']:
            assert offset is None
            if mode == 'random_per_class': 
                self.offset = np.random.choice(range(len(labels) // Nclasses), 
                              Nclasses) # Same (random) offset within a class
        elif mode == 'fixed': # Same offset for every image
            self.offset = [offset] * Nclassess
        else:
            self.offset = offset # Same offset within a class

    def __call__(self, data):
        im, target = data

        # Get a class specific background (either random, or consistent
        # for all images of that class)
        class_backgrounds = np.where(self.labels == target)[0]
        if self.offset is None:
            bidx = np.random.choice(class_backgrounds, 1)[0]
        else:
            bidx = class_backgrounds[self.offset[target]]

        bg = F.interpolate(self.backgrounds[bidx:bidx+1].clone(), 
                        size=im.size)
        bg = transforms.functional.to_pil_image(bg[0], mode='RGB')
        bg.putalpha(255)
        return Image.alpha_composite(bg, im)

def details_df(root: Path, early_stop: int = -1):
    with open(root / 'details.log') as f:
        lines = f.readlines()
        if early_stop < 0: early_stop = len(lines)
        lines = zip(range(early_stop), lines)
        infos = [flatten(json.loads(l), reducer='path') for _, l in lines]
    infos = [x for x in infos if osp.exists(root / f"{x[ID_COL]}.png")]
    return pd.DataFrame(infos)


ID_COL = 'image_id'
LABEL_TO_IND = {v: k for (k, v) in CLASS_DICT['CIFAR'].items()}
class ModelDataset(VisionDataset):
    def __init__(self, root, transform, subset=None, early_stop=-1,
                 extra_cols=[ID_COL], model_filter=None, **kwargs):
        if isinstance(root, str):
            self.root = Path(root)
            self.info_df = details_df(self.root, early_stop)
        elif isinstance(root, tuple):
            self.root, self.info_df = root
        else:
            raise ValueError("Unrecognized type for argument 'root'")

        if subset is not None:
            self.info_df = self.info_df.sample(n=subset, random_state=0)
            self.info_df.index = range(len(self.info_df))

        if model_filter is not None:
            self.info_df = self.info_df[self.info_df['uuid'].isin(model_filter)]
            self.info_df.index = range(len(self.info_df))
            
        self.transform = transform
        self.extra_cols = extra_cols
        self.tx_with_lab = kwargs.get('tx_with_lab', True)
    
    def __getitem__(self, ind):
        row = self.info_df.loc[ind]
        img = Image.open(self.root / f"{row[ID_COL]}.png").convert("RGBA")
        target = LABEL_TO_IND[row['class']]

        if self.transform is not None:
            img = self.transform((img, target)) if self.tx_with_lab else self.transform(img)

        extra_args = row[self.extra_cols].tolist()
        return [img, target, *extra_args]
    
    def __len__(self):
        return len(self.info_df)

