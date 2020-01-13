import json
import os
import pickle
import sys
from collections import deque
from datetime import datetime
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sns.set()

# ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ
ON_KAGGLE: bool = 'KAGGLE_URL_BASE' in os.environ

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
CONFIG_DIR = BASE_DIR / 'configs'
DATA_DIR = Path(
    '../input/data-science-bowl-2019/') if ON_KAGGLE else BASE_DIR / 'data' / 'original'
FEATURE_DIR = BASE_DIR / 'features'
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


def dump_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)


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


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def compute_object_size(o, handlers={}):
    def dict_handler(d): return chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    # estimate sizeof object without __sizeof__
    default_size = sys.getsizeof(0)

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def make_activities_map(train, test):
    list_of_user_activities = list(set(train.main_df['title'].value_counts(
    ).index).union(set(test.main_df['title'].value_counts().index)))
    activities_map = dict(
        zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    return activities_map


def make_win_code(activities_map):
    win_code = dict(zip(activities_map.values(),
                        (4100*np.ones(len(activities_map))).astype('int')))
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    return win_code


def get_accuracy_group(true_attempts, false_attempts):
    accuracy = true_attempts / \
        (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
    accuracy_group = None
    if accuracy == 0:
        accuracy_group = 0
    elif accuracy == 1:
        accuracy_group = 3
    elif accuracy == 0.5:
        accuracy_group = 2
    else:
        accuracy_group = 1

    return accuracy_group, accuracy
