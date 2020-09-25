from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import seaborn as sns
from matplotlib import pyplot as plt 
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--csv-path', required=True)
args = parser.parse_args()

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                 'dog', 'frog', 'horse', 'ship', 'truck']
df = pd.read_csv(args.csv_path)
cf_mat = confusion_matrix(df['labs'], df['preds'])
sns.heatmap(cf_mat / cf_mat.sum(), annot=True, 
            fmt='.1%', cmap='Blues')
plt.savefig("cf_mat.png")