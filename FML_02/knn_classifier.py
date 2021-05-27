import numpy as np

from utils import euclidean_norm


def get_knn(k, X_train, X_test):
    """
    Get k-nearest neighbors for each point in the test set
    :param k: int -- Number of neighbors to include in the majority vote
    :param X_train: ndarray (n, d) -- Design matrix
    :param X_test: ndarray (m, d) -- Test matrix
    :return: ndarray (k, m) -- Matrix containing in each column the index of knns' in training set
    """
    # Find k nearest neighbors for each test data point
    dist = euclidean_norm(X_train, X_test)
    # ndarray (k, m) -- containing in each column the idx of knns' in training set.
    idx = np.argsort(dist, axis=0)[:k, :]
    return idx


def knn_classifier(X_train, X_test, y_train, k, C=10):
    """
    Predicts class of a given test set following k-NN heuristics
    :param X_train: ndarray (n, d) -- Design matrix
    :param X_test: ndarray (m, d) -- Test matrix
    :param y_train: ndarray (n, ) -- Outcome matrix of training data
    :param k: int -- Number of neighbors to include in the majority vote
    :param C: int -- Number of classes
    :return: ndarray (m, ) -- Outcome matrix of predictions for test data
    """
    idx = get_knn(k, X_train, X_test)
    # ndarray (k, m) -- containing per column labels of the knn of the training set
    neighbors = np.take(y_train, idx)
    # ndarray (C, m) -- containing votes for each class per test point
    election = np.apply_along_axis(lambda x: np.bincount(x, minlength=C), axis=0, arr=neighbors)
    # ndarray (m, ) -- containing label prediction for test set
    prediction = np.argmax(election, axis=0)
    return prediction


def calculate_error_knn_classifier(X_train, X_test, y_train, y_test, k):
    """

    :param X_train: ndarray (n, d) -- Design matrix
    :param X_test: ndarray (m, d) -- Test matrix
    :param y_test: ndarray (m, ) -- Outcome matrix of training data
    :param k: int -- Number of neighbors to include in the majority vote
    :return: int -- Out-of-sample error
    """
    batch_size = X_test.shape[0]
    predictions = knn_classifier(X_train, X_test, y_train, k)
    n_errors = np.count_nonzero(y_test != predictions)
    ose = 100 * n_errors / batch_size
    return ose


def knn_compare(class_A, class_B, X_train, X_test, y_train, y_test, k):
    """
    Filter classes in data set and compute ose error between classes
    :param class_A: int -- Class to be filtered in data set
    :param class_B: int -- Class to be filtered in data set
    :param X_train: ndarray (n, d) -- Design matrix
    :param X_test: ndarray (m, d) -- Test matrix
    :param y_train: ndarray (n, ) -- Outcome matrix of training data
    :param y_test: ndarray (m, ) -- Outcome matrix of test data
    :param k: int -- Number of neighbors to include in the majority vote
    :return: int -- Out-of-sample error
    """
    X_train_AB = X_train[np.logical_or(class_A == y_train, class_B == y_train)]
    y_train_AB = y_train[np.logical_or(class_A == y_train, class_B == y_train)]

    X_test_AB = X_test[np.logical_or(class_A == y_test, class_B == y_test)]
    y_test_AB = y_test[np.logical_or(class_A == y_test, class_B == y_test)]

    ose = calculate_error_knn_classifier(X_train_AB, X_test_AB, y_train_AB, y_test_AB, k)
    return ose


def compute_confusion_matrix(X_train, X_test, y_train, y_test, K, C=10):
    """
    Compute confusion matrix taking into consideration all classes
    :param K: ndarray (k, ) -- Array with different k's to participate in the majority vote
    :param C: int -- Number of classes
    :return: ndarray (len(k), C, C) -- Confusion matrix
    """
    confusion = np.empty((len(K), C, C))
    for k in range(len(K)):
        for i in range(C):
            for j in range(C):
                confusion[k, i, j] = knn_compare(i, j, X_train, X_test, y_train, y_test, K[k])
    return confusion
