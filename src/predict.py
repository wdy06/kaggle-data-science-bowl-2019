import argparse
import os
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import features
import metrics
import mylogger
import preprocess
import utils
from pathlib import Path
from dataset import DSB2019Dataset
from optimizedrounder import OptimizedRounder
from runner import Runner

parser = argparse.ArgumentParser(description='kaggle data science bowl 2019')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--input-dir", type=str, help="input directory path")
args = parser.parse_args()

print(f'on kaggle: {utils.ON_KAGGLE}')
print(utils.DATA_DIR)
print(utils.RESULTS_BASE_DIR)


# result_dir = utils.RESULTS_BASE_DIR / \
#     utils.make_experiment_name(args.debug)
# os.mkdir(result_dir)

# logger = mylogger.get_mylogger(filename=result_dir / 'log')
# logger.debug(f'created: {result_dir}')
# logger.debug('loading data ...')
input_dir = Path(args.input_dir)
print(f'input dir: {input_dir}')
print('loading data ...')

# train = DSB2019Dataset(mode='train', debug=args.debug)
# test = DSB2019Dataset(mode='test')

# logger.debug('preprocessing ...')
# train = preprocess.preprocess_dataset(train)
# test = preprocess.preprocess_dataset(test)
# encode title
# activities_map = utils.load_json(utils.CONFIG_DIR / 'activities_map.json')
# train.main_df['title'] = train.main_df['title'].map(activities_map)
# test.main_df['title'] = test.main_df['title'].map(activities_map)
# train.train_labels['title'] = train.train_labels['title'].map(
#     activities_map)

# win_code = utils.make_win_code(activities_map)

# train.main_df['timestamp'] = pd.to_datetime(train.main_df['timestamp'])
# test.main_df['timestamp'] = pd.to_datetime(test.main_df['timestamp'])

# new_train = features.generate_features(
#     train.main_df, win_code, mode='train')
train_feat_path = utils.FEATURE_DIR / 'train_features.pkl'
test_feat_path = utils.FEATURE_DIR / 'test_features.pkl'
if args.debug:
    train_feat_path = utils.FEATURE_DIR / 'train_features_debug.pkl'
    test_feat_path = utils.FEATURE_DIR / 'test_features_debug.pkl'

# new_train = utils.load_pickle(train_feat_path)

features_list = utils.load_yaml(input_dir / 'features_list.yml')
all_features = features_list['features']
print(all_features)
# X, y = new_train[all_features], new_train['accuracy_group']

config_path = input_dir / 'model_config.yml'
config = utils.load_yaml(config_path)
print(config)
model_params = config['model_params']
model_params['categorical_feature'] = features_list['categorical_features']


# oof = np.zeros(len(X))
# NFOLDS = 5
# folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

# training_start_time = time()
# fold_indices = list(folds.split(X, y))
runner = Runner(run_name='train_cv',
                x=None,
                y=None,
                model_cls=config['model_class'],
                params=model_params,
                metrics=metrics.qwk,
                save_dir=input_dir,
                fold_indices=None
                )

# process test set
if utils.ON_KAGGLE:
    test = DSB2019Dataset(mode='test')
    test = preprocess.preprocess_dataset(test)
    activities_map = utils.load_json(utils.CONFIG_DIR / 'activities_map.json')
    win_code = utils.make_win_code(activities_map)
    X_test = features.generate_features_by_acc(
        test.main_df, win_code, mode='test')
else:
    X_test = utils.load_pickle(test_feat_path)

preds = runner.run_predict_all(X_test[all_features])
# preds = runner.run_predict_cv(X_test[all_features])
if config['task'] == 'regression':
    optR = OptimizedRounder()
    best_coef = utils.load_pickle(input_dir / 'best_coef.pkl')
    preds = optR.predict(preds, best_coef)
# save_path = result_dir / f'submission_val{val_score:.5f}.csv'
if utils.ON_KAGGLE:
    save_path = 'submission.csv'
    submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
    submission['accuracy_group'] = np.round(preds).astype('int')
    submission.to_csv(save_path, index=False)
print('finish !!!')
