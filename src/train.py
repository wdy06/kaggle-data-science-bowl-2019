import argparse
import os
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import features
import metrics
import mylogger
import utils
from dataset import DSB2019Dataset
from models.model_lgbm import ModelLGBMClassifier
from runner import Runner

parser = argparse.ArgumentParser(description='kaggle data science bowl 2019')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

print(f'on kaggle: {utils.ON_KAGGLE}')
print(utils.DATA_DIR)
print(utils.RESULTS_BASE_DIR)

try:

    result_dir = utils.RESULTS_BASE_DIR / \
        utils.make_experiment_name(args.debug)
    # slack.notify_start(experiment_name)
    os.mkdir(result_dir)

    logger = mylogger.get_mylogger(filename=result_dir / 'log')
    logger.debug(f'created: {result_dir}')
    logger.debug('loading data ...')

    train = DSB2019Dataset(mode='train', debug=args.debug)
    test = DSB2019Dataset(mode='test')

    logger.debug('preprocessing ...')
    # encode title
    activities_map = utils.load_json(utils.CONFIG_DIR / 'activities_map.json')
    train.main_df['title'] = train.main_df['title'].map(activities_map)
    test.main_df['title'] = test.main_df['title'].map(activities_map)
    train.train_labels['title'] = train.train_labels['title'].map(
        activities_map)

    win_code = utils.make_win_code(activities_map)

    train.main_df['timestamp'] = pd.to_datetime(train.main_df['timestamp'])
    test.main_df['timestamp'] = pd.to_datetime(test.main_df['timestamp'])

    new_train = features.generate_features(
        train.main_df, win_code, mode='train')

    features_list = utils.load_yaml(utils.CONFIG_DIR / 'features_list.yml')
    all_features = features_list['features']
    logger.debug(all_features)
    X, y = new_train[all_features], new_train['accuracy_group']

    config_path = utils.CONFIG_DIR / '000_lgbm_baseline.yml'
    config = utils.load_yaml(config_path)
    model_params = config['model_params']
    model_params['categorical_feature'] = features_list['categorical_features']

    if not utils.ON_KAGGLE:
        model_params['device'] = 'gpu'
        model_params['gpu_platform_id'] = 0
        model_params['gpu_device_id'] = 0

    oof = np.zeros(len(X))
    NFOLDS = 5
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

    training_start_time = time()
    fold_indices = list(folds.split(X, y))
    runner = Runner(run_name='train_cv',
                    x=X[all_features],
                    y=y,
                    model_cls=config['model_class'],
                    params=model_params,
                    metrics=metrics.qwk,
                    save_dir=result_dir,
                    fold_indices=fold_indices
                    )
    val_score = runner.run_train_cv()

    logger.debug('-' * 30)
    logger.debug(f'OOF QWK: {val_score}')
    logger.debug('-' * 30)

    # process test set
    X_test = features.generate_features(test.main_df, win_code, mode='test')
    runner.run_train_all()
    preds = runner.run_predict_all(X_test[all_features])

    save_path = result_dir / f'submission_val{val_score:.5f}.csv'
    if utils.ON_KAGGLE:
        save_path = 'submission.csv'
    submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
    submission['accuracy_group'] = np.round(preds).astype('int')
    submission.to_csv(save_path, index=False)
    logger.debug(f'save to {save_path}')


except Exception as e:
    print(e)
    logger.exception(e)
    raise e
