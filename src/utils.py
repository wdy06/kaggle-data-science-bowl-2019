from sklearn.metrics import confusion_matrix
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ
ON_KAGGLE: bool = 'KAGGLE_URL_BASE' in os.environ

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = Path(
    '../input/data-science-bowl-2019/') if ON_KAGGLE else BASE_DIR / 'data' / 'original'
RESULTS_BASE_DIR = Path('.') if ON_KAGGLE else BASE_DIR / 'results'
# RESULTS_BASE_DIR = BASE_DIR / 'results'


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_feature_importance(model, columns, path):
    df = pd.DataFrame()
    df['importance'] = np.log(model.feature_importances_)
    df.index = columns
    df.sort_values(by='importance', ascending=True, inplace=True)
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(y=df.index, width=df.importance)
    plt.savefig(path)


def qwk(act, pred, n=4, hist_range=(0, 3)):

    O = confusion_matrix(act, pred)
    O = np.divide(O, np.sum(O))

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i-j)**2)/((n-1)**2)

    act_hist = np.histogram(act, bins=n, range=hist_range)[0]
    prd_hist = np.histogram(pred, bins=n, range=hist_range)[0]

    E = np.outer(act_hist, prd_hist)
    E = np.divide(E, np.sum(E))

    num = np.sum(np.multiply(W, O))
    den = np.sum(np.multiply(W, E))

    return 1-np.divide(num, den)
