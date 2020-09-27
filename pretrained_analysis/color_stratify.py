from argparse import ArgumentParser
from make_predictions import make_predictions
from torchvision import transforms
from robustness import datasets, model_utils
from utils import RGBAToRGB, ModelDataset, details_df
from colorsys import hsv_to_rgb, rgb_to_hsv
from os import path
from pathlib import Path
from objectnet_utils import OBJN_TO_IN_MAP
from itertools import product, cycle

import torch as ch
import pandas as pd
import numpy as np

# For parallelism
import multiprocessing as mp
from multiprocessing import Pool

# For plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
from IPython import embed

def vis_stratification(results_path, mode=['hue', 'value']):
    df = pd.read_csv(results_path)
    all_hsv = ['hue', 'saturation', 'value']
    f_col = list(set(all_hsv) - set(mode))[0]

    # 3D Plot
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')

    # Make accuracy dataframe
    df['sameclass'] = (df['preds'] == df['labs'])

    acc_df = df.groupby(all_hsv + ['sameclass']).agg({'num': 'sum'}).reset_index()
    acc_df = acc_df.pivot(index=all_hsv, columns='sameclass', values='num').reset_index()
    acc_df['acc'] = acc_df[True] / (acc_df[True] + acc_df[False])
    del acc_df[True], acc_df[False]

    perclass_acc_df = df.groupby(all_hsv + ['sameclass', 'labs']).agg({'num': 'sum'}).reset_index()
    perclass_acc_df = perclass_acc_df.pivot(index=all_hsv + ['labs'], columns='sameclass', values='num').reset_index()
    perclass_acc_df['acc'] = perclass_acc_df[True] / (perclass_acc_df[True] + perclass_acc_df[False])
    perclass_acc_df = perclass_acc_df.loc[perclass_acc_df.groupby('labs')['acc'].idxmax()]
    print(perclass_acc_df)
    embed()

    for f_val in acc_df[f_col].unique():
        _df = acc_df[(acc_df[f_col] - f_val).abs() < 1e-2]
        hsvs = _df.apply(lambda r: hsv_to_rgb(float(r['hue']), 
                                            float(r['saturation']), 
                                            float(r['value'])), axis=1)
        ax3D.scatter(*[_df[x] for x in mode], _df['acc'], s=30, c=hsvs, marker='o')     
        fig.savefig(f"color_stratify_results/3d_stratify_{f_col}_{f_val:.2f}.png")
        plt.close()

        cl_mat = _df.pivot(index=mode[0], columns=mode[1], values='acc')
        sns.heatmap(cl_mat, annot=True, fmt='.0%', cmap='Blues')

        plt.tight_layout()
        plt.savefig(f"color_stratify_results/2d_stratify_{f_col}_{f_val:.2f}.png")
        plt.close()

def eval_hsv(data):
    cfgs, device, ds = data
    with ch.cuda.device(device):
        m, _ = model_utils.make_and_restore_model(arch='resnet50', 
                                                  dataset=ds,
                                                  pytorch_pretrained=True)
        m.eval().cuda()

    all_res = []
    for h, s, v in cfgs:
        ds.transform_test = transforms.Compose([
            RGBAToRGB(hsv_to_rgb(h, s, v)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        with ch.cuda.device(device):
            res = make_predictions(m, ds, OBJN_TO_IN_MAP,
                                   mode='restrict',
                                   workers=0, batch_size=100)

        res['hue'] = h
        res['saturation'] = s
        res['value'] = v
        #grp_keys = ['labs', 'preds', 'hue', 'saturation', 'value']
        all_res.append(res)
    return pd.concat(all_res) #.groupby(grp_keys).agg(num=('uids', 'count')).reset_index()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path')
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--model-filter')
    parser.add_argument('--num-examples', type=int, default=10000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--granularity', type=int, default=10)
    args = parser.parse_args()

    if path.exists(args.out_path):
        print("Path already exists, only visualizing...")
        vis_stratification(args.out_path)
        sys.exit(0)
    else:
        assert args.dataset_path is not None

    print("Preparing dataset...")
    ds_path = Path(args.dataset_path)
    root_df = details_df(ds_path)
    ds = datasets.ImageNet((ds_path, root_df))

    model_filter = args.model_filter and ch.load(args.model_filter)

    ds.custom_class = ModelDataset
    ds.custom_class_args = {
        'subset': args.num_examples,
        #'model_filter': model_filter
    }

    all_dfs = []
    prev_transforms = ds.transform_test
    cfgs = product(*[np.linspace(0, 1, args.granularity + 1) for _ in range(3)])
    cfgs = np.array_split(list(cfgs), args.threads)
    cfgs = zip(cfgs, cycle(range(ch.cuda.device_count())), cycle([ds]))
    
    print("Start multi...")
    mp.set_start_method('spawn') 
    with Pool(args.threads) as p:
        all_dfs = p.map(eval_hsv, cfgs)
    print(f"Done {len(all_dfs)} models...")

    final_df = pd.concat(all_dfs)
    final_df.to_csv(args.out_path)