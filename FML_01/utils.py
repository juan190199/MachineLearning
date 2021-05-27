import numpy as np


def create_data(N):
    """
    Create data using inverse transform method

    :param N: int
        Batch size of the data to be created

    :return: array-like of shape (N, N)
        Data set with N instances
    """
    Y = np.random.randint(0, 2, size=N)  # Sample instance labels from prior 1/2
    if N == 2:
        while np.all(Y == Y[0]):
            Y = np.random.randint(0, 2, size=N)  # Sample instance labels from prior 1/2

    u = np.random.uniform(size=N)
    X = np.empty(N)
    idx0 = Y == 0
    idx1 = ~idx0
    X[idx0] = 1 - np.sqrt(1 - u[idx0])
    X[idx1] = np.sqrt(u[idx1])
    data_set = np.stack((X, Y), axis=1)
    return data_set

