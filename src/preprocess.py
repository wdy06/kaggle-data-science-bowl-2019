import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import utils


def preprocess_dataset(dataset):
    # encode title
    activities_map = utils.load_json(utils.CONFIG_DIR / 'activities_map.json')
    world_map = utils.load_json(utils.CONFIG_DIR / 'world_map.json')
    dataset.main_df['title'] = dataset.main_df['title'].map(activities_map)
    dataset.main_df['world'] = dataset.main_df['world'].map(world_map)

    dataset.main_df['timestamp'] = pd.to_datetime(dataset.main_df['timestamp'])

    dataset.main_df.sort_values(['installation_id', 'timestamp'], inplace=True)

    dataset.main_df['end_of_game'] = dataset.main_df['game_session'].ne(
        dataset.main_df['game_session'].shift(-1).ffill())

    return dataset


def find_high_corr_feature(df, threshold):
    count = 0
    to_remove = []
    for feat_a in df.columns:
        for feat_b in df.columns:
            if (feat_a != feat_b) and (feat_a not in to_remove) and (feat_b not in to_remove):
                corr = np.corrcoef(df[feat_a], df[feat_b])[0][1]
                if corr > threshold:
                    to_remove.append(feat_b)
                    count += 1
                    print(
                        f'{count}: FEAT_A: {feat_a} FEAT_B: {feat_b} - Correlation: {corr}')
    print('finished find high corr feature')
    return to_remove


def stract_hists(feature, train, test, adjust=False, plot=False):
    n_bins = 10
    train_data = train[feature]
    test_data = test[feature]
    if adjust:
        test_data *= train_data.mean() / test_data.mean()
    perc_90 = np.percentile(train_data, 95)
    train_data = np.clip(train_data, 0, perc_90)
    test_data = np.clip(test_data, 0, perc_90)
    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)
    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)
    msre = mean_squared_error(train_hist, test_hist)
    if plot:
        print(msre)
        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)
        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)
        plt.show()
    return msre


def adjust_data(features_list, categorical_feat, train, test):
    to_exclude = []
    adjust_dict = {}
    ajusted_test = test.copy()
    white_list = [
        'ave_accuracy',
        'ave_accuracy_group',
        'title12_ave_accracy',
        'title12_ave_accracy_group',
        'title12_last_acc_group',
        'title18_ave_accracy',
        'title18_ave_accracy_group',
        'title18_last_acc_group',
        'title21_ave_accracy',
        'title21_ave_accracy_group',
        'title21_last_acc_group',
        'title2_ave_accracy',
        'title2_ave_accracy_group',
        'title2_last_acc_group',
        'title30_ave_accracy',
        'title30_ave_accracy_group',
        'title30_last_acc_group'
    ] + categorical_feat
    for feature in [feat for feat in features_list if feat not in white_list]:
        # print(feature)
        # data = train[feature]
        train_mean = train[feature].mean()
        # data = test[feature]
        test_mean = test[feature].mean()
        if test_mean == 0:
            print(feature, train_mean, test_mean)
            to_exclude.append(feature)
            continue
        # error = stract_hists(feature, train, test, adjust=True)
        ajust_factor = train_mean / test_mean
        if ajust_factor > 10 or ajust_factor < 0.1:  # or error > 0.01:
            to_exclude.append(feature)
            # print(feature, train_mean, test_mean, error, ajust_factor)
            print(feature, train_mean, test_mean, ajust_factor)
        else:
            adjust_dict[feature] = ajust_factor
            ajusted_test[feature] *= ajust_factor
        # except Exception as e:
        #     print(e)
        #     to_exclude.append(feature)
        #     print(feature, train_mean, test_mean)

    return train, test, ajusted_test, to_exclude, adjust_dict


def preprocess_for_nn(x_train, x_test, y_train, features_list, cat_feat_list):
    encoder_dict = {}
    new_train = np.array([])
    new_test = np.array([])
    print(new_train.shape)
    if len(cat_feat_list) > 0:
        for cat_feat_name in cat_feat_list:
            print(cat_feat_name)
            encoder = OneHotEncoder(sparse=False)
            feat_train = encoder.fit_transform(x_train[[cat_feat_name]])
            feat_test = encoder.transform(x_test[[cat_feat_name]])
            new_train = np.concatenate(
                (new_train, feat_train), axis=1) if new_train.size else feat_train
            new_test = np.concatenate(
                (new_test, feat_test), axis=1) if new_test.size else feat_test
            encoder_dict[cat_feat_name] = encoder
    numeric_feat_list = [
        feat for feat in features_list if feat not in cat_feat_list]
    scaler = MinMaxScaler()
    feat_train = scaler.fit_transform(x_train[numeric_feat_list])
    feat_test = scaler.transform(x_test[numeric_feat_list])
    new_train = np.concatenate([new_train, feat_train], axis=1)
    new_test = np.concatenate([new_test, feat_test], axis=1)
    encoder_dict['numeric'] = scaler

    y_scaler = MinMaxScaler()
    new_train_y = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    encoder_dict['accuracy_group'] = y_scaler
    new_train = pd.DataFrame(new_train)
    new_test = pd.DataFrame(new_test)
    new_train_y = pd.DataFrame(new_train_y)

    return new_train, new_test, new_train_y, encoder_dict


def preprocess_for_nn_from_encoder_dict(x_test, features_list, cat_feat_list, encoder_dict):
    new_test = np.array([])
    if len(cat_feat_list) > 0:
        for cat_feat_name in cat_feat_list:
            print(cat_feat_name)
            encoder = encoder_dict[cat_feat_name]
            feat_test = encoder.transform(x_test[[cat_feat_name]])
            new_test = np.concatenate(
                (new_test, feat_test), axis=1) if new_test.size else feat_test
    numeric_feat_list = [
        feat for feat in features_list if feat not in cat_feat_list]
    scaler = encoder_dict['numeric']
    feat_test = scaler.transform(x_test[numeric_feat_list])
    new_test = np.concatenate([new_test, feat_test], axis=1)
    new_test = pd.DataFrame(new_test)

    return new_test


def postprocess_for_nn(preds_y, encoder_dict, true_y=None):

    preds_y = encoder_dict['accuracy_group'].inverse_transform(
        preds_y.reshape(-1, 1))
    preds_y = preds_y.flatten()

    if true_y is not None:
        true_y = encoder_dict['accuracy_group'].inverse_transform(true_y)
        true_y = true_y.flatten()
        return preds_y, true_y
    else:
        return preds_y
