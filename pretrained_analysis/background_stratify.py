from argparse import ArgumentParser
from make_predictions import make_predictions
from torchvision import transforms
from robustness import datasets, model_utils
from utils import RGBAToRGBWithBackground, ModelDataset, details_df
from colorsys import hsv_to_rgb, rgb_to_hsv
from os import path
from pathlib import Path

import os
import torch as ch
import pandas as pd

# For plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
from IPython import embed

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/scratch/datasets/SIMFAR-10/v1.2')
    parser.add_argument('--background-path', type=str, default='/data/theory/robustopt/shibani/synthfar/')
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--num-examples', type=int, default=2000)
    parser.add_argument('--background-mode', type=str, default='random_per_class')
    parser.add_argument('--num-grid-samples', type=int, default=1500)
    parser.add_argument('--model-filter', type=str, default='good_models.pt')
    args = parser.parse_args()

    if path.exists(os.path.join(args.out_path, f'{args.background_mode}.csv')):
        print("Path already exists...")
        sys.exit(0)
    else:
        assert args.dataset_path is not None
        assert args.background_path is not None

    print("Loading dataset and backgrounds...")

    ds_path = Path(args.dataset_path)
    root_df = details_df(ds_path)
    ds = datasets.ImageNet((ds_path, root_df))

    b_path = Path(args.background_path)
    bimages = ch.load(f'{b_path}/background_images.pt')
    blabels = ch.load(f'{b_path}/background_labels.pt')

    model_filter = args.model_filter and ch.load(args.model_filter)

    ds.custom_class = ModelDataset
    ds.custom_class_args = {
        'subset': args.num_examples,
        'model_filter': model_filter,
        'random_labels': True
    }

    print("Loading model...")

    m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
            resume_path='/data/theory/robustopt/robust_models/cifar_nat/checkpoint.pt.best')
    m.eval()
    m = ch.nn.DataParallel(m)

    print("Starting data generation...")

    all_dfs = []
    prev_transforms = ds.transform_test

    if args.background_mode in ['random', 'random_per_class']:
        for i in range(args.num_grid_samples):
            background_transform = RGBAToRGBWithBackground(bimages, 
                                            blabels,
                                            mode=args.background_mode,
                                            offset=None)
            ds.transform_test = transforms.Compose([
                    background_transform,
                    transforms.CenterCrop((40, 40)),
                    transforms.Resize(32),
                    transforms.ToTensor()
                ])
            res = make_predictions(m, ds)
            res['offset'] = [str(background_transform.offset)] * len(res['labs'])
            res['mode'] = [args.background_mode] * len(res['labs'])
            grp_keys = ['labs', 'preds', 'mode', 'offset']
            all_dfs.append(res.groupby(grp_keys).agg(num=('uids', 'count')).reset_index())
            print(f"Dsone {len(all_dfs)} models...")
    else:
        print("Not implemented yet")
        sys.exit(0)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    final_df = pd.concat(all_dfs)
    final_df.to_csv(os.path.join(args.out_path, f'{args.background_mode}.csv'))