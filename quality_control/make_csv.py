import sys
sys.path.append('../pretrained_analysis/')
from utils import ModelDataset, details_df
from robustness import datasets
import pandas as pd 
from argparse import ArgumentParser
from pathlib import Path
from IPython import embed
from os import path as osp

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--ims-per-model', type=int, default=8)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--exclude', help='location of output csv to exclude')
    args = parser.parse_args()
    assert not osp.exists(args.out_path)

    N = args.ims_per_model 
    info_df = details_df(Path(args.dataset_path))
    info_df.groupby('uuid')
    ten_imgs_per_model = info_df.groupby(['uuid', 'class']).sample(n=N)
    grouped = ten_imgs_per_model.groupby(['class', 'uuid']).agg({'image_id': list})
    formatted = grouped.image_id.apply(pd.Series)
    formatted.columns = [f'image-{i}' for i in range(N)]
    formatted = formatted.reset_index()
    
    print(f"DataFrame length: {formatted['uuid'].count()}")
    if args.exclude is not None:
        to_exclude = pd.read_csv(args.exclude)
        formatted = formatted[~formatted['uuid'].isin(to_exclude['uuid'])]
        print(f"New DataFrame length: {formatted['uuid'].count()}")

    formatted.to_csv(args.out_path, index=False)