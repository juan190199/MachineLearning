from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import numpy as np


def data_preparation(digits, filter=None, test_size=0.33, random_seed=12345):
    """

    :param digits: dict
        Dictionary containing data set.
        digits['data'] contains all features of design matrix
        digits['target'] contains respective target labels

    :param filter: array-like of shape (n, ), default=None
        Array with targets to be filters from data

    :param test_size: int, default=0.33
        Percentage of test data after splitting

    :param random_seed: int, default=12345
        Random seed to make results reproducible

    :return:
    """
    data = digits['data']
    target = digits['target']

    # Data filtering
    if filter:
        filters = [target == f for f in filter]
        mask = np.sum(filters, axis=0)
        data = data[np.argwhere(mask)]/data.max()
        target = target[np.where(mask)]

        # Relable targets
        for i, f in enumerate(filter):
            target[target == f] = i

    # Random split
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_seed)

    train_std = np.std(X_train, axis=0) + 1e-99
    train_mean = np.mean(X_train, axis=0)
    # Standardize training data
    X_train_std = (X_train - train_mean)/train_std
    # Standardize test data
    X_test_std = (X_test - train_mean)/train_std

    return X_train, X_train_std, X_test, X_test_std, y_train, y_test
