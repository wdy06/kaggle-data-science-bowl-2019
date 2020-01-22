import argparse
import os
import shutil
from time import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import features
import metrics
import mylogger
import preprocess
import utils
from dataset import DSB2019Dataset
from optimizedrounder import OptimizedRounder
from runner import Runner

parser = argparse.ArgumentParser(description='kaggle data science bowl 2019')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

print(f'on kaggle: {utils.ON_KAGGLE}')
print(utils.DATA_DIR)
print(utils.RESULTS_BASE_DIR)

try:
    N_TRIALS = 500
    if args.debug:
        N_TRIALS = 2

    result_dir = utils.RESULTS_BASE_DIR / \
        utils.make_experiment_name(args.debug)
    os.mkdir(result_dir)

    logger = mylogger.get_mylogger(filename=result_dir / 'log')
    logger.debug(f'created: {result_dir}')
    logger.debug('loading data ...')

    train_feat_path = utils.FEATURE_DIR / 'train_features.pkl'
    test_feat_path = utils.FEATURE_DIR / 'test_features.pkl'
    if args.debug:
        train_feat_path = utils.FEATURE_DIR / 'train_features_debug.pkl'
        test_feat_path = utils.FEATURE_DIR / 'test_features_debug.pkl'

    new_train = utils.load_pickle(train_feat_path)

    features_list = utils.load_yaml(utils.CONFIG_DIR / '508_features_list.yml')
    all_features = features_list['features']
    if args.debug:
        all_features = [
            feat for feat in all_features if feat in new_train.columns]
    logger.debug(all_features)
    X, y = new_train[all_features], new_train['accuracy_group']

    oof = np.zeros(len(X))
    # NFOLDS = 5
    new_train.reset_index(inplace=True)
    fold_indices = []
    for i_fold in new_train.fold.unique():
        train_idx = new_train.index[new_train['fold'] != i_fold].tolist()
        val_idx = new_train.index[new_train['fold'] == i_fold].tolist()
        fold_indices.append((train_idx, val_idx))
        print(len(train_idx), len(val_idx))
    # fold_indices = list(folds.split(X, y))

    def objective(trial):
        # model_cls = 'ModelLGBMRegressor'
        model_cls = 'ModelCatBoostRegressor'
        # params_spaces = {
        #     'verbosity': -1,
        #     'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        #     'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        #     'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        #     'max_depth': trial.suggest_int('max_depth', 3, 12),
        #     'num_leaves': trial.suggest_int('num_leaves', 15, 511),
        #     'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        #     'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        # }
        # if params_spaces['boosting_type'] != 'goss':
        #     params_spaces['bagging_fraction'] = trial.suggest_uniform(
        #         'bagging_fraction', 0.4, 1.0)
        #     params_spaces['bagging_freq'] = trial.suggest_int(
        #         'bagging_freq', 1, 7)
        params_spaces = {
            'thread_count': -1,
            'n_estimators': 10000,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': -1,
            'early_stopping_rounds': 300,
            # 'task_type': 'GPU',
            # 'devices': '0',
            'depth': trial.suggest_int('max_depth', 3, 12),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 100),
            'border_count': trial.suggest_int('border_count', 32, 200),
        }
        runner = Runner(run_name='train_cv',
                        x=X[all_features],
                        y=y,
                        model_cls=model_cls,
                        params=params_spaces,
                        metrics=metrics.rmse,
                        save_dir=result_dir,
                        fold_indices=fold_indices
                        )
        val_score, oof_preds = runner.run_train_cv()
        return val_score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    save_result_path = result_dir / 'study.pkl'
    utils.dump_pickle(study, save_result_path)
    logger.debug(f'save to {save_result_path}')

except Exception as e:
    print(e)
    logger.exception(e)
    raise e
