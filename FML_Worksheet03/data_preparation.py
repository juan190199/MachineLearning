import numpy as np


def data_filtering(digits, seed=12345, test_percentage=0.3, filter=None, split=True):
    """
    Preparation of the data: Filtering, and relabeling of targets
    :param digits: dict -- Dictionary containing data set
    :param test_percentage: int -- Percentage of test instances
    :param seed: int -- Fixed seed for random shuffle
    :param filter: list -- List of numbers to be filtered
    :param split: bool, default=True --
    If True, data is split in a training and test sets. Otherwise, no split is done
    :return: tuple -- data[train_idx], data[test_idx], target[train_idx], target[test_idx]
    """
    data = digits['data']
    target = digits['target']

    # Data filtering
    if filter:
        # Boolean list of length len(filter). Each element of this list is another list containing boolean variables
        # to distinguish if the number should be filtered or not
        filters = [target == f for f in filter]
        mask = np.sum(filters, axis=0)
        # print(np.argwhere(mask == 1))
        data = data[np.where(mask)]/data.max()
        target = target[np.where(mask)]

        # Relable targets
        for i, f in enumerate(filter):
            target[target == f] = i

    # Random split
    if split:
        X_train, X_test, y_train, y_test = data_split(data, target, test_percentage, seed=seed)
        return X_train, X_test, y_train, y_test
    else: return data, target


def data_split(data, target, test_percentage=0.3, seed=12345):
    """
    Data split given a percentage of test instances
    :param data: ndarray(n, 2) -- Data matrix
    :param target: ndarray (n, ) -- Outcome labels of data
    :param test_percentage: int -- Percentage of test instances
    :param seed: int -- Fixed seed for random shuffle
    :return: tuple -- data[train_idx], data[test_idx], target[train_idx], target[test_idx]
    """
    # Calculate test size
    test_size = int(data.shape[0] * test_percentage)

    # Shuffle modifies indices inplace
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    # A fixed seed and a fixed series of calls to ‘RandomState’ methods using the same parameters
    # will always produce the same results
    rstate = np.random.RandomState(seed)
    rstate.shuffle(indices)

    train_idx = indices[test_size:]
    test_idx = indices[:test_size]

    return data[train_idx], data[test_idx], target[train_idx], target[test_idx]


def reduce_dimension(X):
    """
    Reduce dimension of data to 2d
    :param X: ndarray (n, d) -- d dimensional data
    :return: ndarray (n, 2) -- 2 dimensional data
    """
    # mean(Upper part) - mean(Lower part)
    feat1 = (np.mean(X[:, :X.shape[-1] // 4], axis=-1) -
             np.mean(X[:, 3 * X.shape[-1] // 4:], axis=-1))

    # mean(Upper part) * mean(Lower part)
    feat2 = (np.mean(X[:, :X.shape[-1] // 4] * X[:, 3 * X.shape[-1] // 4:], axis=-1))

    return np.array([feat1, feat2]).T


def worse_reduce_dimension(X):
    """
    Reduce dimension of data to 2d
    :param X: ndarray (n, d) -- d dimensional data
    :return: ndarray (n, 2) -- 2 dimensional data
    """
    # mean(Image)
    feat1 = np.mean(X, axis=-1)

    # var(Image)
    feat2 = np.var(X, axis=-1)

    return np.array([feat1, feat2]).T
