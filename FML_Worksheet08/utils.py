import numpy as np
from numpy.linalg import lstsq


def orthogonal_matching_pursuit(X, y, iterations):
    """

    :param X:
    :param y:
    :param iterations: int
    :return:

    """
    dim = X.shape[1]
    S = []
    res = y
    feat_idx = list(range(X.shape[1]))
    theta_hat = np.empty((dim, iterations))
    for it in range(iterations):
        # Correlations with the current residual
        cor = [np.abs(np.dot(X[:, j].T, res)) for j in feat_idx]
        # Find maximum
        j_max = np.argmax(np.array(cor))
        # Add most important feature to selected set S
        S.append((feat_idx.pop(j_max)))
        # Update the residual with the projection by solving least squares problem
        new_X = X[:, S]
        theta = lstsq(new_X, y, rcond=None)
        res = y - np.dot(new_X, theta[0])
        theta_hat[S, it] = theta[0]

    return theta_hat
