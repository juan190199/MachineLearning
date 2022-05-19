import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


def accuracy_score(y, y_pred):
    accuracy = np.sum(y == y_pred, axis=0)
    return accuracy


class Loss(object):
    def loss(self, y, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        return NotImplementedError()

    def acc(self, y, y_pred):
        return NotImplementedError()


class Zero_One_Loss():
    def loss(self, y, y_pred):
        n_samples = len(y)
        accuracy = accuracy_score(y, y_pred)
        return n_samples - accuracy

    def gradient(self, y, y_pred):
        return 1
