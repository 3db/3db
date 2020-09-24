import json
import torch as ch
from pathlib import Path
import pandas as pd
from robustness import datasets, model_utils
from tqdm import tqdm
from os import path
from argparse import ArgumentParser
from utils import ModelDataset, RGBAToRGB
from torchvision import transforms

def make_predictions(model, ds, map_dict, mode='restrict',
                     workers=10, batch_size=500):
    _, loader = ds.make_loaders(workers=workers, 
                                batch_size=batch_size, 
                                only_val=True)
    
    in_classes = list(set(map_dict.values()))
    
    dfs = []
    with ch.no_grad():
        for ims, labs, uids in tqdm(loader, total=len(loader)):
            ims = ims[:,:3,...].cuda() # RGBA -> RGB
            op = model(ims)[0]
            if mode != 'restrict':
                preds = ch.argmax(op, dim=1).cpu()
            else:
                op = op[:, in_classes]
                preds = ch.argmax(op, dim=1).cpu()
                preds = ch.tensor([in_classes[int(p)] for p in preds])
            
            labs = ch.tensor([map_dict[int(l)] for l in labs.cpu()])
            dfs.append(pd.DataFrame({'uids': uids,
                                    'preds': preds.cpu().numpy(),
                                    'labs': labs.numpy()}))

    return pd.concat(dfs)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--model-filter')
    args = parser.parse_args()

    assert not path.exists(args.out_path), "out-path must not already exist"

    model_filter = args.model_filter and ch.load(args.model_filter)
    ds = datasets.CIFAR(args.dataset_path)
    ds.custom_class = ModelDataset
    ds.transform_test = transforms.Compose([
        RGBAToRGB((1.0, 1.0, 1.0)),
        transforms.CenterCrop((40, 40)),
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    ds.custom_class_args = {
        'model_filter': model_filter
    }

    m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
            resume_path='/data/theory/robustopt/robust_models/cifar_nat/checkpoint.pt.best')
    m.eval()
    m = ch.nn.DataParallel(m)

    final_df = make_predictions(m, ds)
    final_df['correct'] = (final_df['preds'] == final_df['labs']).astype(float)
    print(f"Accuracy: {100 * final_df['correct'].mean():.2f}")
    print(f"Mean-Per-Class Accuracy: {100 * final_df.groupby('labs').agg({'correct': 'mean'})['correct'].mean():.2f}")
    final_df.to_csv(args.out_path)
