import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold

from sklearn.utils.validation import _assert_all_finite, _num_samples

from data_preparation import reduce_dimension


def euclidean_norm(x, y):
    """
    Computes euclidean norm between two ndarrays
    :param x: ndarray (n, d) -- ndarray with data points
    :param y: ndarray (m, d) -- ndarray with data points
    :return: ndarray (n, m) -- Euclidean norm between all data points
    """
    if len(x.shape) == 1:
        x = x[None, :]
    if len(y.shape) == 1:
        y = y[None, :]

    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).squeeze()


def calculate_error(y_pred, y_test):
    """
    Calculate out-of-sample error of the classifier for a data set
    :param y_pred: ndarray (m, ) -- Vector of predictions
    :param y_test: ndarray (m, ) -- Test matrix
    :return:
    """
    return np.mean(y_pred == y_test)


def kf_cross_validation(data, target, model, n_splits=10):
    """

    :param data:
    :param target:
    :return:
    """
    kf = KFold(n_splits=n_splits)
    mean_rate = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        Xr_train, Xr_test = reduce_dimension(data[train_index]), reduce_dimension(data[test_index])
        model.fit(Xr_train, target[train_index])
        pred = model.predict(Xr_test)
        mean_rate.append(np.mean(pred == target[test_index]))

    print("Model {model}. Mean Accuracy Cross Validation: {mean:.2f} +/- {std:.2f}".format(model=model,
                                                                                           mean=np.mean(mean_rate),
                                                                                           std=np.std(mean_rate)))


def logsumexp(arr, axis=0):
    """
    Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of over/underflow.
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
