import numpy as np
from numpy.linalg import lstsq


def orthogonal_matching_pursuit(X, y, iterations):
    """
    For every iteration, calculates the optimal weight vector with D dimensions

    :param X: ndarray of shape (n_instances, n_features)
        Design matrix with n_instances instances and dimension n_features

    :param y: ndarray of shape (n_instances, )
        Output array for n_instances instances

    :param iterations: int
        Number of iterations for calculation of optimal weight vector
        ToDo: Implement OMP with termination criteria

    :return: ndarray of shape (n_features, iterations)
        Optimal weight vector for each iteration
    """
    dim = X.shape[1]
    theta_hat = np.zeros((dim, iterations))
    S = []
    feat_idx = list(range(X.shape[1]))
    res = y
    for it in range(iterations):
        # Correlations with the current residual
        cor = [np.abs(np.dot(X[:, j].T, res)) for j in feat_idx]
        # Find maximum
        j_max = np.argmax(np.array(cor))
        # Add most important feature to selected set S
        S.append(feat_idx.pop(j_max))
        # Update the residual with the projection by solving least squares problem
        X_c = X[:, S]
        theta = lstsq(X_c, y, rcond=None)
        res = y - np.dot(X_c, theta[0])
        theta_hat[S, it] = theta[0]

    return theta_hat


def pred_acc(X_test, theta, y_test):
    """

    :param X_test: ndarray of shape (n_instances, n_features)
        Test matrix

    :param theta: ndarray of shape (n_features, )
        Weight vector

    :param y_test: ndarray of shape (n_instances, )
        Output/target vector

    :return: int
        Accuracy of predictions
    """
    pred = np.dot(X_test, theta)
    pred[np.where(pred < 0)] = 0
    pred[np.where(pred > 0)] = 1
    return np.mean(pred == y_test)
