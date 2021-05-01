from utils import create_data

import numpy as np


def threshold_classifier(type, X=None, threshold=None, error=False):
    """
    :param type: string -- Classifier type
    :param X: ndarray (n, ) -- Numpy array with test set to be classified
    :param threshold: int -- Threshold of the classifier
    :param error: Boolean variable.
    If True, calculate analytical error of the classifier; otherwise empirical
    :return:
    If error is false, ndarray (n, ) -- Predictions for test set.
    Otherwise, int -- Analytical error for the given classifier type and threshold
    """
    if type == 'A':
        if error is True:
            return 100 * (1 / 4 + (threshold - 1 / 2) ** 2)
        else:
            binary_arr = np.where(X < threshold, 0, 1)
            return binary_arr
    if type == 'B':
        if error is True:
            return 100 * (3 / 4 - (threshold - 1 / 2) ** 2)
        else:
            binary_arr = np.where(X < threshold, 1, 0)
            return binary_arr
    if type == 'C':
        if error is True:
            return 100 * 1 / 2
        else:
            return np.random.randint(0, 2, len(X))
    if type == 'D':
        if error is True:
            return 100 * 1 / 2
        else:
            return np.ones(len(X))


def error_threshold_classifier(type, analytical=False, batch=None, n_data_sets=None, threshold=None):
    """
    Calculate out-of-sample error (generalization error) given number of data sets
    :param type: string -- Classifier type
    :param analytical: Boolean variable.
    If True, calculate analytical error of the classifier; otherwise empirical
    :param batch: int -- Size of the test set
    :param n_data_sets: int -- Number of data sets to be evaluated
    :param threshold: int -- Threshold of the classifier
    :return:
    If analytical is False, ndarray (n_data_sets,) -- Out of sample error for different training sets
    Otherwise, int -- Analytical error
    """
    if analytical is False:
        oses = np.empty(n_data_sets)
        for i in range(n_data_sets):
            test_set = create_data(batch)
            prediction = threshold_classifier(type=type, X=test_set[:, 0], threshold=threshold)
            n_errors = np.sum(np.abs(np.subtract(prediction, test_set[:, 1])))
            ose = n_errors * 100 / batch
            oses[i] = ose
        return oses
    else:
        a_error = threshold_classifier(type=type, threshold=threshold, error=True)
        return a_error


def nn_classifier(training_set, test_set):
    """
    Calculate decision rule following nearest neighbors rule
    :param training_set: ndarray (n, k) -- Training set of n instances and k labels
    :param test_set: ndarray (n, k) -- Test set of n instances and k labels
    :return: ndarray (test_set.shape[0], ) -- Prediction following nearest neighbor rule
    """
    prediction = np.empty(test_set.shape[0])
    for i in range(test_set.shape[0]):
        diff = np.abs(training_set[:, 0] - test_set[i, 0])
        idx = np.argmin(diff)
        prediction[i] = training_set[idx, 1]
    return prediction


def error_nn_classifier(size_data, batch_size, n_data_sets):
    """
    Compute out of sample error of nearest neighbor classifier
    :param size_data: int -- size of the training set
    :param batch_size: int -- Size of the test set
    :param n_data_sets: int -- Number of data sets to be evaluated
    :return: ndarray (n_data_sets,) -- Out of sample error for different training sets
    """
    test_set = create_data(batch_size)
    oses = np.empty(n_data_sets)
    for i in range(n_data_sets):
        data = create_data(size_data)
        prediction = nn_classifier(data, test_set)
        n_errors = np.sum(np.abs(np.subtract(prediction, test_set[:, 1])))
        ose = 100 * n_errors / batch_size
        oses[i] = ose
    return oses
