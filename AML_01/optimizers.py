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


def stochastic_gradient_descent(w, X, y, alpha0, momentum=None, gamma=None, n_iterations=10, test=False):
    for t in range(n_iterations):
        idx = np.random.choice(y.shape[0], size=1, replace=False)
        alpha = alpha0 / (1 + gamma * t)
        w -= alpha * loss_gradient(w, X[idx], y[idx])
        if test:
            train_loss['sgd'].append(zero_one_loss(predict(w, X_train), y_train))
            test_loss['sgd'].append(zero_one_loss(predict(w, X_test), y_test))
        return w


def stochastic_gradient_minibatch(w, X, y, alpha0, momentum=None, gamma=None, n_iterations=10, batch_size=20, test=False):
    for t in range(n_iterations):
        batch_idx = np.random.choice(y.shape[0], size=batch_size, replace=False)
        alpha = alpha0 / (1 + gamma * t)
        w -= alpha * loss_gradient(w, X[batch], y[batch])
        if test:
            train_loss['sgb'].append(zero_one_loss(predict(w, X_train), y_train))
            test_loss['sgb'].append(zero_one_loss(predict(w, X_test), y_test))
        return w


def stochastic_gradient_momentum(w, X, y, alpha0, momentum=None, gamma=None, n_iterations=10, test=False):
    w_update = np.zeros(w.shape)
    for t in range(n_iterations):
        idx = np.random.choice(y.shape[0], size=1, replace=False)
        alpha = alpha0 / (1 + gamma * t)
        w_update = momentum * w_update + (1 - momentum) * loss_gradient(w, X[idx], y[idx])
        w -= alpha * w_update
        if test:
            train_loss['sgm'].append(zero_one_loss(predict(w, X_train), y_train))
            test_loss['sgm'].append(zero_one_loss(predict(w, X_test), y_test))
        return w



def ADAM():
    ...


def stochastic_average_gradient():
    ...


def dual_coordinate_ascent():
    ...


def newton_raphson():
    ...
