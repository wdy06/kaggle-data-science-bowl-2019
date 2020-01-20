import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

import features
import metrics
import preprocess
import utils
from dataset import DSB2019Dataset
from optimizedrounder import HistBaseRounder
from weightoptimzer import WeightOptimzer
from runner import Runner

parser = argparse.ArgumentParser(description='kaggle data science bowl 2019')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--optimize", help="auto tune ensemble weight",
                    action="store_true")
parser.add_argument("--test", help="run test mode",
                    action="store_true")
parser.add_argument("--config", "-c", type=str, help="ensemble config path")
args = parser.parse_args()

print(f'on kaggle: {utils.ON_KAGGLE}')

result_dict = []
# train_feat_path = utils.FEATURE_DIR / 'train_features.pkl'
test_feat_path = utils.FEATURE_DIR / 'test_features.pkl'
all_test_feat_path = utils.FEATURE_DIR / 'all_test_features.pkl'
if args.debug:
    # train_feat_path = utils.FEATURE_DIR / 'train_features_debug.pkl'
    test_feat_path = utils.FEATURE_DIR / 'test_features_debug.pkl'
    all_test_feat_path = utils.FEATURE_DIR / 'all_test_features_debug.pkl'

train = DSB2019Dataset(mode='train')
event_code_list = list(train.main_df.event_code.unique())
event_id_list = list(train.main_df.event_id.unique())

del train
gc.collect()

# process test set
activities_map = utils.load_json(
    utils.CONFIG_DIR / 'activities_map.json')
win_code = utils.make_win_code(activities_map)
if utils.ON_KAGGLE:
    test = DSB2019Dataset(mode='test')
    test = preprocess.preprocess_dataset(test)
    X_test_org, all_test_history = features.generate_features_by_acc(
        test.main_df, win_code, event_code_list, event_id_list, mode='test')
    del test
    gc.collect()
else:
    X_test_org = utils.load_pickle(test_feat_path)
    all_test_history = utils.load_pickle(all_test_feat_path)

X_test_org = features.add_feature(X_test_org, activities_map)
X_test_org = features.add_agg_feature_test(X_test_org, all_test_history)

ens_test_preds = np.zeros(X_test_org.shape[0])

ens_config = utils.load_yaml(args.config)

sum_weight = 0
preds_df = pd.DataFrame()
test_preds_df = pd.DataFrame()
for i, one_config in enumerate(ens_config):
    # print('-'*30)
    # print(f'{i}: {one_config}')
    input_dir = utils.RESULTS_BASE_DIR / one_config['exp_name']
    if utils.ON_KAGGLE:
        input_dir = Path('/kaggle/input/') / one_config['exp_name']
    config = utils.load_yaml(input_dir / 'model_config.yml')
    X_train = utils.load_pickle(input_dir / 'train_x.pkl')
    y_train = utils.load_pickle(input_dir / 'train_y.pkl')
    fold_indices = utils.load_pickle(input_dir / 'fold_indices.pkl')
    model_params = config['model_params']
    runner = Runner(run_name='train_cv',
                    x=X_train,
                    y=y_train,
                    model_cls=config['model_class'],
                    params=model_params,
                    metrics=metrics.qwk,
                    save_dir=input_dir,
                    fold_indices=fold_indices
                    )
    oof_preds, true_y = runner.get_oof_preds()
    preds_df[i] = oof_preds
    if config['model_class'] == 'ModelNNRegressor':
        encoder_dict = utils.load_pickle(input_dir / 'encoder_dict.pkl')
        oof_preds, true_y = preprocess.postprocess_for_nn(
            oof_preds, encoder_dict, true_y)

    weight = one_config['weight']
    sum_weight += weight
    if i < 1:
        ens_oof_preds = oof_preds * weight
    else:
        ens_oof_preds += oof_preds * weight

    if config['task'] == 'regression':
        val_rmse = metrics.rmse(oof_preds, true_y)
        optR = HistBaseRounder()
        best_coef = utils.load_pickle(input_dir / 'best_coef.pkl')
        print(f'best threshold: {best_coef}')
        oof_preds = optR.predict(oof_preds, best_coef)
        val_score = metrics.qwk(oof_preds, true_y)
        print(f'rmse: {val_rmse}')
    print(f'qwk: {val_score}')
    result_dict.append({'exp_name': one_config['exp_name'],
                        'model_name': one_config['model_name'],
                        'weight': weight,
                        'val_rmse': val_rmse,
                        'val_qwk': val_score}
                       )

    # predict test
    X_test = X_test_org.copy()
    features_list = utils.load_yaml(input_dir / 'features_list.yml')
    all_features = features_list['features']
    cat_features = features_list['categorical_features']
    # adjust data
    if os.path.exists(input_dir / 'adjust.json'):
        print('adjust !!!')
        adjust_dict = utils.load_json(input_dir / 'adjust.json')
        for key, factor in adjust_dict.items():
            # print(f'{key}: {factor}')
            X_test[key] *= factor
    X_test = X_test[all_features]
    if config['model_class'] == 'ModelNNRegressor':
        print('preprocessing for nn ...')
        # encoder_dict = utils.load_pickle(input_dir / 'encoder_dict.pkl')
        X_test = preprocess.preprocess_for_nn_from_encoder_dict(
            X_test, all_features, cat_features, encoder_dict)
    test_preds = runner.run_predict_cv(X_test)
    if config['model_class'] == 'ModelNNRegressor':
        print('post processing for nn ...')
        test_preds = preprocess.postprocess_for_nn(test_preds, encoder_dict)
    test_preds_df[i] = test_preds
    ens_test_preds += test_preds * weight


for one_result in result_dict:
    print('-'*30)
    print(f'exp name: {one_result["exp_name"]}')
    print(f'model name: {one_result["model_name"]}')
    print(f'weight: {one_result["weight"]}')
    print(f'val rmse: {one_result["val_rmse"]}')
    print(f'val qwk: {one_result["val_qwk"]}')

# find best coef
print('-'*30)
ens_oof_preds /= sum_weight
if config['task'] == 'regression':
    val_rmse = metrics.rmse(ens_oof_preds, true_y)
    optR = HistBaseRounder()
    optR.fit(ens_oof_preds, true_y)
    ens_best_coef = optR.coefficients()
    print(f'ensemble best threshold: {ens_best_coef}')
    ens_oof_preds = optR.predict(ens_oof_preds, ens_best_coef)
    val_score = metrics.qwk(ens_oof_preds, true_y)

print(f'ensemble rmse: {val_rmse}')
print(f'ensemble qwk: {val_score}')

# find best weight
if args.optimize:
    print('finding best weight')
    weihgt_opt = WeightOptimzer(preds_df, true_y)
    _, opt_weight = weihgt_opt.fit()
    # print(f'optimzed score: {-optmized_score}')
    # print(f'optimzed weight: {opt_weight}')
    ens_oof_preds = weihgt_opt.weight_pred(preds_df)
    if config['task'] == 'regression':
        val_rmse = metrics.rmse(ens_oof_preds, true_y)
        optR = HistBaseRounder()
        optR.fit(ens_oof_preds, true_y)
        ens_best_coef = optR.coefficients()
        print(f'optimzed ensemble best threshold: {ens_best_coef}')
        ens_oof_preds = optR.predict(ens_oof_preds, ens_best_coef)
        val_score = metrics.qwk(ens_oof_preds, true_y)
        print(f'optimzed rmse score: {val_rmse}')
        print(f'optimzed qwk score: {val_score}')
        print(f'optimzed weight: {opt_weight}')
        print(f'optimzed best coef: {ens_best_coef}')


ens_test_preds /= sum_weight
if args.optimize:
    ens_test_preds = weihgt_opt.weight_pred(test_preds_df)

if config['task'] == 'regression':
    optR = HistBaseRounder()
    # best_coef = utils.load_pickle(input_dir / 'best_coef.pkl')
    ens_test_preds = optR.predict(ens_test_preds, ens_best_coef)

if utils.ON_KAGGLE:
    save_path = 'submission.csv'
    submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
    submission['accuracy_group'] = (ens_test_preds).astype('int')
    submission.to_csv(save_path, index=False)
print('finish !!!')
