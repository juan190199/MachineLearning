import numpy as np

from utils import (loss_gradient, zero_one_loss, predict, sigmoid)


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


def stochastic_gradient_minibatch(w, X, y, alpha0, momentum=None, gamma=None, n_iterations=10, batch_size=20,
                                  test=False):
    for t in range(n_iterations):
        batch_idx = np.random.choice(y.shape[0], size=batch_size, replace=False)
        alpha = alpha0 / (1 + gamma * t)
        w -= alpha * loss_gradient(w, X[batch], y[batch])
        if test:
            train_loss['sgb'].append(zero_one_loss(predict(w, X_train), y_train))
            test_loss['sgb'].append(zero_one_loss(predict(w, X_test), y_test))
        return w


def stochastic_gradient_momentum(w, X, y, alpha0, momentum=0.9, gamma=None, n_iterations=10, test=False):
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


def adam(w, X, y, alpha0, momentum1=0.9, momentum2=0.99, gamma=None, n_iterations=10, eps=1e-8, test=False):
    w_update1 = np.zeros(w.shape)
    w_update1 = np.zeros(w.shape)
    for t in range(n_iterations):
        idx = np.random.choice(y.shape[0], size=1, replace=True)
        w_update1 = momentum1 * w_update1 + (1 - momentum1) * loss_gradient(w, X[idx], y[idx])
        w_update2 = momentum2 * w_update2 + (1 - momentum2) * loss_gradient(w, X[idx], y[idx]) ** 2

        # Bias correction
        w_update1 = w_update1 / (1 - momentum1 ** (t + 1))
        w_update2 = w_update2 / (1 - momentum2 ** (t + 1))

        w -= alpha0 / (w_update2 ** 0.5 + eps) * momentum1

        if test:
            train_loss['adam'].append(zero_one_loss(predict(w, X_train), y_train))
            test_loss['adam'].append(zero_one_loss(predict(w, X_test), y_test))
        return w


def stochastic_average_gradient():
    ...


def dual_coordinate_ascent():
    ...


def newton_raphson(w, X, y, alpha0, momentum=0.9, gamma=None, n_iterations=10, lmbd=0, test=False):
    for t in range(n_iterations):
        z = np.dot(X, w)
        y_t = y / sigmoid(y * z)
        W = np.diag(((sigmoid(z) * sigmoid(-z)) * lmbd / X.shape[0]).reshape(-1, ))
        w = (np.linalg.inv(np.eye(X.shape[1]) + np.dot(X.T, W).dot(X))).dot(X.T).dot(W).dot(z + y_t)

        if test:
            train_loss['nr'].append(zero_one_loss(predict(w, X_train), y_train))
            test_loss['nr'].append(zero_one_loss(predict(w, X_test), y_test))
        return w