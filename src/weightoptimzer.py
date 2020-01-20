
import numpy as np
import scipy as sp

from metrics import qwk
from optimizedrounder import HistBaseRounder


class WeightOptimzer:
    def __init__(self, preds_df, true_y):
        super().__init__()
        self.preds_df = preds_df
        self.true_y = true_y
        self.num_ensemble = self.preds_df.shape[1]
        self.weight = np.ones(self.num_ensemble) / self.num_ensemble

    def _loss_func(self, weight):
        ens_preds = np.zeros(self.preds_df.shape[0])
        for i, col in enumerate(self.preds_df.columns):
            ens_preds += self.preds_df[col].values * weight[i]

        optR = HistBaseRounder()
        optR.fit(ens_preds, self.true_y)
        coef = optR.coefficients()
        ens_preds = optR.predict(ens_preds, coef)
        score = qwk(ens_preds, self.true_y)

        return -score

    def fit(self):
        result = sp.optimize.minimize(self._loss_func,
                                      self.weight,
                                      constraints=(
                                          {'type': 'eq', 'fun': lambda w: 1-sum(w)}),
                                      method='Nelder-Mead',  # 'SLSQP',
                                      bounds=[(0.0, 1.0)] * self.num_ensemble)

        self.weight = result['x']
        return result['fun'], self.weight

    def weight_pred(self, test_preds_df):
        ens_preds = np.zeros(test_preds_df.shape[0])
        for i, col in enumerate(test_preds_df.columns):
            ens_preds += test_preds_df[col].values * self.weight[i]

        return ens_preds
