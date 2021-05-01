import numpy as np

from utils import (euclidean_norm, calculate_error)


def nearest_mean_classifier(X_train, X_test, y_train,  error=False, y_test=None):
    """
    Calculates means for different classes given training data, and also predicts for test data
    :param X_train: ndarray (n, d) -- Design matrix
    :param X_test: ndarray (m, d) -- Test matrix
    :param y_train: ndarray (n, ) -- Outcome matrix of training data
    :return: tuple --
    predictions: ndarray (m, ) -- Vector of predictions
    means: ndarray (labels.shape[0], X_train.shape[1]) -- Means of different classes
    """
    labels = np.unique(y_train)
    means = np.empty(shape=(labels.shape[0], X_train.shape[1]))
    # Calculate mean of all labels
    for i, label in enumerate(labels):
        means[label, :] = np.mean(X_train[y_train == label], axis=0)

    dist = np.empty(shape=(X_test.shape[0], len(labels)))
    for i, label in enumerate(labels):
        dist[:, i] = euclidean_norm(X_test, means[i, :]).squeeze()

    predictions = np.argmin(dist, axis=-1)

    if error:
        return calculate_error(predictions, y_test)
    else:
        return predictions, means



