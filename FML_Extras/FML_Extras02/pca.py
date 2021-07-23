import numpy as np


def pca(X, n_dim):
    """

    :param X: ndarray of shape (n_samples, n_features)
        Data

    :param n_dim: int
        Target dimension

    :return: ndarray of shape (n_samples, n_dim)

    """
    X = X - np.mean(X, axis=0, keepdims=True)
    print(X)
    cov = np.dot(X.T, X)

    eigvals, eigvecs = np.linalg.eig(cov)
    idxs_ = np.argsort(-eigvals)[:n_dim]
    principal_eigvalues = eigvals[idxs_]
    principal_eigvecs = eigvecs[:, idxs_]

    ndim_X = np.dot(X, principal_eigvecs)
    return ndim_X


def highdim_pca(X, n_dim):
    """

    :param X:
    :param n_features:
    :return:
    """
    n = X.shape[0]
    X = X - np.mean(X, axis=0, keepdims=True)

    ncov = np.dot(X, X.T)

    neigvals, neigvecs = np.linalg.eig(ncov)
    idxs_ = np.argsort(-neigvals)[:n_dim]
    principal_neigvals = neigvals[idxs_]
    principal_neigvecs = neigvecs[:, idxs_]

    principal_eigvecs = np.dot(X.T, principal_neigvecs)
    principal_eigvecs = principal_eigvecs / (n * principal_neigvals.reshape(-1, n_dim)) ** 0.5

    ndim_X = np.dot(X, principal_eigvecs)
    return ndim_X
