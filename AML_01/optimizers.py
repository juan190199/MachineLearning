import numpy as np
from sympy import *


def gradient_descent(w, grad_wrt_w, learning_rate=0.01, momentum=0, n_iterations=100):
    w_updt = np.zeros(np.shape(w))
    for t in range(n_iterations):
        w_updt = momentum * w_updt + (1 - momentum) * grad_wrt_w
        w -= learning_rate * w_updt

    return w


def adagrad(w, grad_wrt_w, learning_rate=0.01, n_iterations=100):
    G = np.zeros(np.shape(w))
    eps = 1e-8
    for t in range(n_iterations):
        G += np.power(grad_wrt_w, 2)
        w -= learning_rate * grad_wrt_w / np.sqrt(G + eps)

    return w


def adadelta(w, grad_wrt_w, rho=0.95, eps=1e-6, n_iterations=100):
    w_updt = np.zeros(np.shape(w))
    E_w_updt = np.zeros(np.shape(w))
    E_grad = np.zeros(np.shape(grad_wrt_w))
    for t in range(n_iterations):
        E_grad = rho * E_grad + (1 - rho) * np.power(grad_wrt_w, 2)

        RMS_delta_w = np.sqrt(E_w_updt + eps)
        RMS_grad = np.sqrt(E_grad + eps)

        adaptative_lr = RMS_delta_w / RMS_grad

        w_update = adaptative_lr * grad_wrt_w
        E_w_updt = rho * E_w_updt + (1 - rho) * np.power(w_update, 2)

        w -= w_updt

    return w


def rmsprop():
    ...


def adam():
    ...


def nesterov_accelerated_gradient():
    ...