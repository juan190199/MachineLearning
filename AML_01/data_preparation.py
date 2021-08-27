import numpy as np


def data_preparation(digits, filter=None):
    """
    Prepare data with given filters

    :param digits: dict
        Dictionary containing data set.
        digits['data'] contains all features of design matrix
        digits['target'] contains respective target labels

    :param filter: array-like of shape (n, ), default=None
        Array with targets to be filters from data

    :return:
    """
    data = digits['data']
    target = digits['target']

    # Data filtering
    if filter:
        # List of n_filters ndarrays wit boolean variables for each filter
        filters = [target == f for f in filter]
        mask = np.sum(filters, axis=0)
        data = data[np.argwhere(mask)].squeeze()
        target = target[np.where(mask)]

        # Relable targets
        targets = [-1, 1]
        for i, f in enumerate(filter):
            target[target == f] = targets[i]

    return data, target
