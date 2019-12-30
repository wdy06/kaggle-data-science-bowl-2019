import pandas as pd

import utils


class DSB2019Dataset():
    def __init__(self, mode='train', debug=False):
        self.mode = mode
        self.debug = debug
        self.main_df = pd.read_csv(utils.DATA_DIR / f'{self.mode}.csv')
        if self.mode == 'train':
            self.train_labels = pd.read_csv(
                utils.DATA_DIR / 'train_labels.csv')
            self.specs = pd.read_csv(utils.DATA_DIR / 'specs.csv')
        if self.debug:
            self.main_df = self.main_df[:40000]
