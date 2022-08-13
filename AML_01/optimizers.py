import numpy as np

from utils import (loss_gradient, zero_one_loss, predict)

# ToDo
def gradient_descent(w, X, y, alpha0, mu=None, gamma=None, n_iterations=10, test=False):
    for t in range(n_iterations):
        w -= alpha0 * loss_gradient(w, X, y)
        if test:
            train_loss['gd'].append(zero_one_loss(predict(w, X_train), y_train))
            test_loss['gd'].append(zero_one_loss(predict(w, X_test), y_test))
        return w


def stochastic_gradient_descent():
    ...


def stochastic_gradient_minibatch():
    ...


def stochastic_gradient_momentum():
    ...


def ADAM():
    ...


def stochastic_average_gradient():
    ...


def dual_coordinate_ascent():
    ...


def newton_raphson():
    ...
