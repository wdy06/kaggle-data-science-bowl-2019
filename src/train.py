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

    features_list = utils.load_yaml(utils.CONFIG_DIR / '501_features_list.yml')
    utils.dump_yaml(features_list, result_dir / 'features_list.yml')
    all_features = features_list['features']
    logger.debug(all_features)
    X, y = new_train[all_features], new_train['accuracy_group']

    config_path = utils.CONFIG_DIR / '001_lgbm_regression.yml'
    config = utils.load_yaml(config_path)
    logger.debug(config)
    utils.dump_yaml(config, result_dir / 'model_config.yml')
    model_params = config['model_params']
    model_params['categorical_feature'] = features_list['categorical_features']

    if not utils.ON_KAGGLE:
        model_params['device'] = 'gpu'
        model_params['gpu_platform_id'] = 0
        model_params['gpu_device_id'] = 0

    oof = np.zeros(len(X))
    # NFOLDS = 5
    # folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
    new_train.reset_index(inplace=True)
    fold_indices = []
    for i_fold in new_train.fold.unique():
        train_idx = new_train.index[new_train['fold'] != i_fold].tolist()
        val_idx = new_train.index[new_train['fold'] == i_fold].tolist()
        fold_indices.append((train_idx, val_idx))
        print(len(train_idx), len(val_idx))
    utils.dump_pickle(fold_indices, result_dir / 'fold_indices.pkl')

    training_start_time = time()
    # fold_indices = list(folds.split(X, y))
    runner = Runner(run_name='train_cv',
                    x=X[all_features],
                    y=y,
                    model_cls=config['model_class'],
                    params=model_params,
                    metrics=metrics.qwk,
                    save_dir=result_dir,
                    fold_indices=fold_indices
                    )
    val_score, oof_preds = runner.run_train_cv()
    if config['task'] == 'regression':
        optR = OptimizedRounder()
        optR.fit(oof_preds, y)
        best_coef = optR.coefficients()
        logger.debug(f'best threshold: {best_coef}')
        utils.dump_pickle(best_coef, result_dir / 'best_coef.pkl')
        oof_preds = optR.predict(oof_preds, best_coef)
        val_score = metrics.qwk(oof_preds, y)

    logger.debug('-' * 30)
    logger.debug(f'OOF QWK: {val_score}')
    logger.debug('-' * 30)

    # process test set
    X_test = utils.load_pickle(test_feat_path)
    runner.run_train_all()
    preds = runner.run_predict_all(X_test[all_features])
    # preds = runner.run_predict_cv(X_test[all_features])
    if config['task'] == 'regression':
        preds = optR.predict(preds, best_coef)
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
