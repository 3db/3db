from argparse import ArgumentParser
from IPython import embed
import pandas as pd 
import torch as ch
import numpy as np

parser = ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--out-path')
args = parser.parse_args()

df = pd.read_csv(args.data)
df['good'] = (df['label'] == 'good').astype(float)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of unique rows: {df['image-1'].unique().shape[0]}")

agg_dict = {
    'score': ('good', 'mean'),
    'count': ('good', 'count')
}
agg_df = df.groupby(['class', 'uuid']).agg(**agg_dict)
num_good, num_bad = [(agg_df['score'] == x).sum() for x in (1.0, 0.0)]
num_undec = agg_df.shape[0] - (num_good + num_bad)

print(f"Model counts | Good: {num_good} "
                    f"| Bad: {num_bad} "
                    f"| Undecided: {num_undec}")

multilabels = (agg_df['count'] > 1).sum()
print(f"{multilabels} Multi-labels ({num_undec / multilabels:.2f} disagreement rate)")

if args.out_path:
    print(f"Saving to {args.out_path}...")
    good_models = set(agg_df[agg_df['score'] == 1.0].reset_index()['uuid'].tolist())
    ch.save(good_models, args.out_path)
    print(f"Saved.")

print('-' * 80)
agg_df['score_c'] = np.floor(agg_df['score'])
agg_dict = {
    'acceptance_rate': ('score_c', 'mean'),
    'num_good': ('score_c', 'sum')
}
perclass_summary = agg_df.reset_index().groupby('class').agg(**agg_dict)
print("Distribution of models over classes:")
print(perclass_summary)