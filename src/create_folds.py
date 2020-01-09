import pandas as pd
from sklearn.model_selection import KFold


def create_folds(df, n_fold):
    kf = KFold(n_splits=5)
    ins_id_list = df.installation_id.unique()
    df['fold'] = -1
    for i, (train_idx, val_idx) in enumerate(kf.split(ins_id_list)):
        print(len(train_idx), len(val_idx))
        train_ins_id = list(ins_id_list[train_idx])
        val_ins_id = list(ins_id_list[val_idx])
        # print(train_ins_id)
        # print(val_ins_id)
        df.loc[df.installation_id.isin(val_ins_id), 'fold'] = i
    return df
