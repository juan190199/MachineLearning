import numpy as np


def sigmoid(x):
    """
    Sigmoid function

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def loss_gradient(w, X, y, lmbd):
    """
    Gradient wrt. w of cross entropy loss with regularization

    :param w:
    :param X:
    :param y:
    :param lmbd:
    :return:
    """
    return -(y - predict(w, X)).dot(X) + w/lmbd


def predict(w, X):
    y_pred = np.dot(X, w)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred < 0] = 0
    return y_pred


def zero_one_loss(y_pred, y):
    """
    Counts the number of wrongly classified samples.

    :param y_pred:

    :param y:

    :return:
    """
    return np.mean(y != y_pred)
