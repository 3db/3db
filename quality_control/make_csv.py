import sys
from utils import ModelDataset, details_df
from robustness import datasets
import pandas as pd 
from argparse import ArgumentParser
from pathlib import Path
from IPython import embed
from os import path as osp
from robustness.tools.vis_tools import show_image_row
from pathlib import Path
from random import shuffle
from os import listdir
from json import load

def example_ims(imgs_path, lab, N=10):
    all_imgs = listdir(imgs_path / 'images' / lab)
    shuffle(all_imgs)
    return pd.Series(all_imgs[:N])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--ims-per-model', type=int, default=8)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--objectnet-path', required=True)
    parser.add_argument('--exclude', help='location of output csv to exclude')
    parser.add_argument('--num-examples', default=5, type=int)
    args = parser.parse_args()
    assert not osp.exists(args.out_path)
    objectnet_path = Path(args.objectnet_path)

    N = args.ims_per_model 
    info_df = details_df(Path(args.dataset_path))
    info_df.groupby('uuid')
    ten_imgs_per_model = info_df.groupby(['uuid', 'class']).sample(n=N)
    grouped = ten_imgs_per_model.groupby(['class', 'uuid']).agg({'image_id': list})
    formatted = grouped.image_id.apply(pd.Series)
    formatted.columns = [f'image-{i}' for i in range(N)]
    formatted = formatted.reset_index()

    class_mapping = load(open('label_map.json'))
    cla_name_fn = lambda x: class_mapping[x]
    formatted['class'] = formatted['class'].apply(cla_name_fn)
    
    N = args.num_examples
    example_im_fn = lambda x: example_ims(Path(args.objectnet_path), x, N)
    ex_col_names = [f'example_im_{i}' for i in range(N)]
    formatted[ex_col_names] = formatted['class'].apply(example_im_fn)
    embed()
    
    print(f"DataFrame length: {formatted['uuid'].count()}")
    if args.exclude is not None:
        to_exclude = pd.read_csv(args.exclude)
        formatted = formatted[~formatted['uuid'].isin(to_exclude['uuid'])]
        print(f"New DataFrame length: {formatted['uuid'].count()}")

    formatted.to_csv(args.out_path, index=False)