import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix

sns.set()

# ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ
ON_KAGGLE: bool = 'KAGGLE_URL_BASE' in os.environ

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
CONFIG_DIR = BASE_DIR / 'configs'
DATA_DIR = Path(
    '../input/data-science-bowl-2019/') if ON_KAGGLE else BASE_DIR / 'data' / 'original'
RESULTS_BASE_DIR = Path('.') if ON_KAGGLE else BASE_DIR / 'results'
# RESULTS_BASE_DIR = BASE_DIR / 'results'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def make_experiment_name(debug):
    experiment_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    if debug:
        experiment_name = 'debug-' + experiment_name
    return experiment_name


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4, cls=NpEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


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


def make_activities_map(train, test):
    list_of_user_activities = list(set(train.main_df['title'].value_counts(
    ).index).union(set(test.main_df['title'].value_counts().index)))
    activities_map = dict(
        zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    return activities_map


def make_win_code(activities_map):
    win_code = dict(zip(activities_map.values(),
                        (4100*np.ones(len(activities_map))).astype('int')))
    win_code[activities_map['Bird Measurer (Assessment)']] = 411
    return win_code
