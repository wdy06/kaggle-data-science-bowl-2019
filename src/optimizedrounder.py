from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import scipy as sp

from metrics import qwk


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5]

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) +
                     [np.inf], labels=[0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = self.coef_
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


class HistBaseRounder:
    def __init__(self):
        super().__init__()
        self.labels = [0, 1, 2, 3]
        self.coef = None

    def fit(self, X, y):
        dist = Counter(y)
        for k in dist:
            dist[k] /= len(y)

        acum = 0
        bound = {}
        for i in range(len(self.labels) - 1):
            acum += dist[i]
            bound[i] = np.percentile(X, acum * 100)
        self.coef = bound

    def predict(self, X, coef=None):
        if not coef:
            coef = self.coef
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3
        return X_p

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef
