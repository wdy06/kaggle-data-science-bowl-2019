import argparse
import os
import shutil
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
from optimizedrounder import HistBaseRounder, OptimizedRounder
from runner import Runner

parser = argparse.ArgumentParser(description='kaggle data science bowl 2019')
parser.add_argument("--config", "-c", type=str,
                    required=True, help="config path")
parser.add_argument("--high-corr", type=str,
                    default=None, help="path to feature list to remove")
parser.add_argument("--adjust", help="adjust train and test hist",
                    action="store_true")
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
    all_test_feat_path = utils.FEATURE_DIR / 'all_test_features.pkl'
    if args.debug:
        train_feat_path = utils.FEATURE_DIR / 'train_features_debug.pkl'
        test_feat_path = utils.FEATURE_DIR / 'test_features_debug.pkl'
        all_test_feat_path = utils.FEATURE_DIR / 'all_test_features_debug.pkl'

    new_train = utils.load_pickle(train_feat_path)
    X_test = utils.load_pickle(test_feat_path)
    X_test_all = utils.load_pickle(all_test_feat_path)

    # some feature engineering
    new_train = features.add_agg_feature_train(new_train)
    X_test = features.add_agg_feature_test(X_test, X_test_all)

    shutil.copyfile(utils.FEATURE_DIR / 'feature_mapper.json',
                    result_dir / 'feature_mapper.json')
    features_list = utils.load_yaml(utils.CONFIG_DIR / '510_features_list.yml')
    utils.dump_yaml(features_list, result_dir / 'features_list.yml')
    all_features = features_list['features']
    categorical_feat = features_list['categorical_features']
    if args.debug:
        all_features = [
            feat for feat in all_features if feat in new_train.columns]

    # remove high corr features
    if args.high_corr:
        logger.debug('remove high corr features')
        remove_feat = utils.load_json(args.high_corr)
        all_features = [
            feat for feat in all_features if feat not in remove_feat]
        features_list['features'] = all_features

    # adjust train and test
    if args.adjust:
        logger.debug('adjusting data ...')
        new_train, _, X_test, to_exclude, adjust_dict = preprocess.adjust_data(
            all_features, categorical_feat, new_train, X_test)
        all_features = [
            feat for feat in all_features if feat not in to_exclude]
        features_list['features'] = all_features
        utils.dump_json(adjust_dict, result_dir / 'adjust.json')
    logger.debug(all_features)
    logger.debug(f'features num: {len(all_features)}')
    utils.dump_yaml(features_list, result_dir / 'features_list.yml')

    X, y = new_train[all_features], new_train['accuracy_group']
    X_test = X_test[all_features]

    config = utils.load_yaml(args.config)
    logger.debug(config)
    utils.dump_yaml(config, result_dir / 'model_config.yml')
    model_params = config['model_params']
    model_params['categorical_feature'] = categorical_feat

    if not utils.ON_KAGGLE:
        if config['model_class'] == 'ModelLGBMRegressor':
            model_params['device'] = 'gpu'
            model_params['gpu_platform_id'] = 0
            model_params['gpu_device_id'] = 0
            model_params['gpu_use_dp'] = True
        if config['model_class'] == 'ModelXGBRegressor':
            model_params['tree_method'] = 'gpu_hist'

    oof = np.zeros(len(X))
    # NFOLDS = 5
    new_train.reset_index(inplace=True)
    fold_indices = []
    for i_fold in new_train.fold.unique():
        train_idx = new_train.index[new_train['fold'] != i_fold].tolist()
        val_idx = new_train.index[new_train['fold'] == i_fold].tolist()
        fold_indices.append((train_idx, val_idx))
        print(len(train_idx), len(val_idx))
    utils.dump_pickle(fold_indices, result_dir / 'fold_indices.pkl')

    runner = Runner(run_name='train_cv',
                    x=X,
                    y=y,
                    model_cls=config['model_class'],
                    params=model_params,
                    metrics=metrics.qwk,
                    save_dir=result_dir,
                    fold_indices=fold_indices
                    )
    val_score, oof_preds = runner.run_train_cv()
    if config['task'] == 'regression':
        val_rmse = metrics.rmse(oof_preds, y)
        # optR = OptimizedRounder()
        optR = HistBaseRounder()
        optR.fit(oof_preds, y)
        best_coef = optR.coefficients()
        logger.debug(f'best threshold: {best_coef}')
        utils.dump_pickle(best_coef, result_dir / 'best_coef.pkl')
        oof_preds = optR.predict(oof_preds, best_coef)
        val_score = metrics.qwk(oof_preds, y)

    logger.debug('-' * 30)
    logger.debug(f'OOF RMSE: {val_rmse}')
    logger.debug(f'OOF QWK: {val_score}')
    logger.debug('-' * 30)

    # process test set
    # X_test = utils.load_pickle(test_feat_path)
    preds = runner.run_predict_cv(X_test)
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
