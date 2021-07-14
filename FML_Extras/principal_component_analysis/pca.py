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

    cov = np.dot(X.T, X)

    eigvals, eigvecs = np.linalg.eig(cov)
    idx_ = np.argsort(eigvals)[:n_dim]
    principal_eigvalues = eigvals[idx_]
    principal_eigvecs = eigvecs[:, idx_]

    ndim_X = np.dot(X, principal_eigvecs)
    return ndim_X
