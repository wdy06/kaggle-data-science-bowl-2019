from sklearn.metrics import confusion_matrix
import numpy as np


def qwk(act, pred, n=4, hist_range=(0, 3)):

    O = confusion_matrix(act, pred)
    O = np.divide(O, np.sum(O))

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i-j)**2)/((n-1)**2)

    act_hist = np.histogram(act, bins=n, range=hist_range)[0]
    prd_hist = np.histogram(pred, bins=n, range=hist_range)[0]

    E = np.outer(act_hist, prd_hist)
    E = np.divide(E, np.sum(E))

    num = np.sum(np.multiply(W, O))
    den = np.sum(np.multiply(W, E))

    return 1-np.divide(num, den)
