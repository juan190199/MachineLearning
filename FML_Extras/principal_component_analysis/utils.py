import numpy as np

from numpy import linalg
from scipy import stats


# Create data
def pdf(x, e):
    return 0.5 * (stats.norm(scale=0.25 / e).pdf(x) + stats.norm(scale=4 / e).pdf(x))


def pca(X):
    """

    :param x:
    :return:
    """
    X = (X - X.mean(axis=0))

    n_data, n_features = X.shape

    if n_features > 100:
        eigvals, eigvects = linalg.eigh(np.dot(X, X.T))
        v = (np.dot(X.T, eigvects).T)[::-1]
        s = np.sqrt(eigvals)[::-1]
    else:
        u, s, v = linalg.svd(X, full_matrices=False)

    return v, s
