import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import accumulators
import utils
from dataset import DSB2019Dataset
import create_folds
import preprocess


def get_data(user_sample, win_code, test_set=False):
    last_activity = 0
    user_activities_count = {'Clip': 0,
                             'Activity': 0, 'Assessment': 0, 'Game': 0}
    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    durations = []
    for i, session in user_sample.groupby('game_session', sort=False):
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        if test_set is True:
            second_condition = True
        else:
            if len(session) > 1:
                second_condition = True
            else:
                second_condition = False

        if (session_type == 'Assessment') & (second_condition):
            all_attempts = session.query(
                f'event_code == {win_code[session_title]}')
            true_attempts = all_attempts['event_data'].str.contains(
                'true').sum()
            false_attempts = all_attempts['event_data'].str.contains(
                'false').sum()
            features = user_activities_count.copy()
            features['session_title'] = session['title'].iloc[0]
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append(
                (session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            features['accumulated_accuracy'] = accumulated_accuracy / \
                counter if counter > 0 else 0
            accuracy = true_attempts / \
                (true_attempts+false_attempts) if (true_attempts +
                                                   false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1

            features.update(accuracy_groups)
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / \
                counter if counter > 0 else 0
            features['accumulated_actions'] = accumulated_actions
            accumulated_accuracy_group += features['accuracy_group']
            accuracy_groups[features['accuracy_group']] += 1
            if test_set is True:
                all_assessments.append(features)
            else:
                if true_attempts+false_attempts > 0:
                    all_assessments.append(features)

            counter += 1

        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type

    if test_set:
        return [all_assessments[-1]]
    return all_assessments


def generate_features(df, win_code, mode):
    if mode == 'train':
        total = 17000
        test_set = False
    elif mode == 'test':
        total = 1000
        test_set = True
    else:
        raise ValueError('mode must be train or test.')
    compiled_data = []
    for i, (ins_id, user_sample) in tqdm(enumerate(df.groupby('installation_id', sort=False)), total=total):
        compiled_data += get_data(user_sample, win_code, test_set)
    return pd.DataFrame(compiled_data)


def generate_features_by_acc(df, win_code, event_code_list, event_id_list, mode):
    if mode == 'train':
        total = 17000
        is_test = False
    elif mode == 'test':
        total = 1000
        is_test = True
    else:
        raise ValueError('mode must be train or test.')
    user_acc = accumulators.UserStatsAcc(
        win_code, event_code_list, event_id_list, is_test)
    # ass_title_acc = accumulators.AssTitleAcc(win_code)
    compiled_feature = []
    all_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(df.groupby('installation_id', sort=False)), total=total):
        user_feature = []
        for k in range(len(user_sample)):
            row = user_sample.iloc[k].to_dict()
            if user_acc.is_labeled_timing(row):
                feature_dict = user_acc.get_stats(row)
                if mode == 'train':
                    # add fold info
                    feature_dict['fold'] = row['fold']
                feature_dict.update(user_acc.assessment_acc.get_stats(row))
                user_feature.append(feature_dict)
            user_acc.update_acc(row)
            # ass_title_acc.update_acc(row)
        if mode == 'train':
            compiled_feature += user_feature
        elif mode == 'test':
            compiled_feature += [user_feature[-1]]
            all_test += user_feature
    compiled_feature = pd.DataFrame(compiled_feature)
    all_test = pd.DataFrame(all_test)
    # feature_mapper = ass_title_acc.get_mapper()
    if mode == 'train':
        return compiled_feature
    else:
        return compiled_feature, all_test


def add_agg_feature_train(df):
    df['ins_session_count'] = df.groupby(
        ['ins_id'])['Clip'].transform('count')
    df['ins_duration_mean'] = df.groupby(
        ['ins_id'])['duration_mean'].transform('mean')
    df['ins_title_nunique'] = df.groupby(
        ['ins_id'])['session_title'].transform('nunique')

    df['sum_event_code_count'] = df[[
        'event_code2050_count', 'event_code4100_count', 'event_code4230_count',
        'event_code5000_count', 'event_code4235_count', 'event_code2060_count',
        'event_code4110_count', 'event_code5010_count', 'event_code2070_count',
        'event_code2075_count', 'event_code2080_count', 'event_code2081_count',
        'event_code2083_count', 'event_code3110_count', 'event_code4010_count',
        'event_code3120_count', 'event_code3121_count', 'event_code4020_count',
        'event_code4021_count', 'event_code4022_count', 'event_code4025_count',
        'event_code4030_count', 'event_code4031_count', 'event_code3010_count',
        'event_code4035_count', 'event_code4040_count', 'event_code3020_count',
        'event_code3021_count', 'event_code4045_count', 'event_code2000_count',
        'event_code4050_count', 'event_code2010_count', 'event_code2020_count',
        'event_code4070_count', 'event_code2025_count', 'event_code2030_count',
        'event_code4080_count', 'event_code2035_count', 'event_code2040_count',
        'event_code4090_count', 'event_code4220_count', 'event_code4095_count']].sum(axis=1)

    df['ins_event_code_count_mean'] = df.groupby(
        ['ins_id'])['sum_event_code_count'].transform('mean')
    return df


def add_agg_feature_test(test, test_all):
    test['ins_session_count'] = test['ins_id'].map(test_all.groupby(
        ['ins_id'])['Clip'].count())
    test['ins_duration_mean'] = test['ins_id'].map(test_all.groupby(
        ['ins_id'])['duration_mean'].mean())
    test['ins_title_nunique'] = test['ins_id'].map(test_all.groupby(
        ['ins_id'])['session_title'].nunique())

    test_all['sum_event_code_count'] = test_all[[
        'event_code2050_count', 'event_code4100_count', 'event_code4230_count',
        'event_code5000_count', 'event_code4235_count', 'event_code2060_count',
        'event_code4110_count', 'event_code5010_count', 'event_code2070_count',
        'event_code2075_count', 'event_code2080_count', 'event_code2081_count',
        'event_code2083_count', 'event_code3110_count', 'event_code4010_count',
        'event_code3120_count', 'event_code3121_count', 'event_code4020_count',
        'event_code4021_count', 'event_code4022_count', 'event_code4025_count',
        'event_code4030_count', 'event_code4031_count', 'event_code3010_count',
        'event_code4035_count', 'event_code4040_count', 'event_code3020_count',
        'event_code3021_count', 'event_code4045_count', 'event_code2000_count',
        'event_code4050_count', 'event_code2010_count', 'event_code2020_count',
        'event_code4070_count', 'event_code2025_count', 'event_code2030_count',
        'event_code4080_count', 'event_code2035_count', 'event_code2040_count',
        'event_code4090_count', 'event_code4220_count', 'event_code4095_count']].sum(axis=1)

    test['ins_event_code_count_mean'] = test['ins_id'].map(test_all.groupby(
        ['ins_id'])['sum_event_code_count'].mean())
    return test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='feature generator')
    parser.add_argument("--debug", help="run debug mode",
                        action="store_true")
    args = parser.parse_args()

    NFOLDS = 5

    if args.debug:
        print('running debug mode ...')
        train_feat_path = utils.FEATURE_DIR / 'train_features_debug.pkl'
        test_feat_path = utils.FEATURE_DIR / 'test_features_debug.pkl'
        all_test_feat_path = utils.FEATURE_DIR / 'all_test_features_debug.pkl'
        feat_mapper_path = utils.FEATURE_DIR / 'feature_mapper_debug.json'
    else:
        train_feat_path = utils.FEATURE_DIR / 'train_features.pkl'
        test_feat_path = utils.FEATURE_DIR / 'test_features.pkl'
        all_test_feat_path = utils.FEATURE_DIR / 'all_test_features.pkl'
        feat_mapper_path = utils.FEATURE_DIR / 'feature_mapper.json'

    print('loading dataset ...')
    train = DSB2019Dataset(mode='train', debug=args.debug)
    test = DSB2019Dataset(mode='test')
    print('preprocessing ...')
    activities_map = utils.load_json(utils.CONFIG_DIR / 'activities_map.json')
    # world_map = utils.load_json(utils.CONFIG_DIR / 'world_map.json')
    train = preprocess.preprocess_dataset(train)
    test = preprocess.preprocess_dataset(test)

    # create folds
    train.main_df = create_folds.create_folds(train.main_df, NFOLDS)

    win_code = utils.make_win_code(activities_map)
    event_code_list = list(train.main_df.event_code.unique())
    event_id_list = list(train.main_df.event_id.unique())

    if not os.path.exists(utils.FEATURE_DIR):
        os.makedirs(utils.FEATURE_DIR)

    train_feature = generate_features_by_acc(
        train.main_df, win_code, event_code_list, event_id_list, mode='train')
    # for feat_name in feature_mapper.keys():
    #     train_feature[feat_name] = train_feature['session_title'].map(
    #         feature_mapper[feat_name])
    print(f'train shape: {train_feature.shape}')

    utils.dump_pickle(train_feature, train_feat_path)
    # utils.dump_json(feature_mapper, feat_mapper_path)

    test_feature, all_test_history = generate_features_by_acc(
        test.main_df, win_code, event_code_list, event_id_list, mode='test')

    # for feat_name in feature_mapper.keys():
    #     test_feature[feat_name] = test_feature['session_title'].map(
    #         feature_mapper[feat_name])
    print(f'test shape: {test_feature.shape}')
    utils.dump_pickle(test_feature, test_feat_path)
    print(f'all test shape: {all_test_history.shape}')
    utils.dump_pickle(all_test_history, all_test_feat_path)

    print('save features !')
    print('finish !!!')
