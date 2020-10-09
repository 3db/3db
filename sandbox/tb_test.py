from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from IPython import embed
from path import Path
from flatten_dict import flatten
from os import path 

writer = SummaryWriter()

ID_COL = 'id'
def details_df(root: Path, early_stop: int = -1):
    with open(root / 'details.log') as f:
        lines = f.readlines()
        if early_stop < 0: early_stop = len(lines)
        lines = zip(range(early_stop), lines)
        infos = [flatten(json.loads(l), reducer='path') for _, l in lines]
    infos = [x for x in infos if path.exists(root/ 'images' / f"{x[ID_COL]}.png")]
    return pd.DataFrame(infos)


df = details_df(root=Path('logs/'))
embed()

