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

    compiled_data = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.main_df.groupby('installation_id', sort=False)), total=17000):
        compiled_data += features.get_data(user_sample, win_code)

    new_train = pd.DataFrame(compiled_data)
    # del compiled_data

    all_features = [
        x for x in new_train.columns if x not in ['accuracy_group']]
    cat_features = ['session_title']
    X, y = new_train[all_features], new_train['accuracy_group']

    config_path = utils.CONFIG_DIR / '000_lgbm_baseline.yml'
    default_param = utils.load_yaml(config_path)

    if not utils.ON_KAGGLE:
        default_param['device'] = 'gpu'
        default_param['gpu_platform_id'] = 0
        default_param['gpu_device_id'] = 0

    oof = np.zeros(len(X))
    NFOLDS = 5
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

    training_start_time = time()
    for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
        start_time = time()
        logger.debug(f'Training on fold {fold+1}')
        clf = LGBMClassifier(**default_param)
        clf.fit(X.loc[trn_idx, all_features], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),
                verbose=100, early_stopping_rounds=100,
                categorical_feature=cat_features)

        oof[test_idx] = clf.predict(
            X.loc[test_idx, all_features], num_iteration=clf.best_iteration_).reshape(len(test_idx))

        logger.debug('Fold {} finished in {}'.format(
            fold + 1, str(timedelta(seconds=time() - start_time))))

    val_score = metrics.qwk(y, oof)
    logger.debug('-' * 30)
    logger.debug(f'OOF QWK: {val_score}')
    logger.debug('-' * 30)

    # train model on all data once
    clf = LGBMClassifier(**default_param)
    clf.fit(X, y, verbose=100, categorical_feature=cat_features)

    # process test set
    new_test = []
    for ins_id, user_sample in tqdm(test.main_df.groupby('installation_id', sort=False), total=1000):
        a = features.get_data(user_sample, win_code, test_set=True)
        new_test.append(a)

    X_test = pd.DataFrame(new_test)
    preds = clf.predict(X_test[all_features],
                        num_iteration=clf.best_iteration_)

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
