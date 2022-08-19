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


def adadelta(w, grad_wrt_w, rho=0.9, eps=1e-8, n_iterations=100):
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


def rmsprop(w, grad_wrt_w, learning_rate=0.01, rho=0.95, eps=1e-6, n_iterations=100):
    Eg = np.zeros(np.shape(grad_wrt_w))
    for t in range(n_iterations):
        Eg = rho * Eg + (1 - rho) * np.power(grad_wrt_w, 2)

        w -= learning_rate * grad_wrt_w / np.sqrt(Eg + eps)

    return w


def adam(w, grad_wrt_w, learning_rate=0.01, b1=0.9, b2=0.999, eps=1e-8, n_iterations=100):
    m = np.zeros(np.shape(grad_wrt_w))
    v = np.zeros(np.shape(grad_wrt_w))

    for t in range(n_iterations):
        m = b1 * m + (1 - b1) * grad_wrt_w
        v = b2 * v + (1 - b2) * np.power(grad_wrt_w, 2)

        m_hat = m / (1 - b1)
        v_hat = v / (1 - b2)

        w_updt = learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        w -= w_updt

    return w


def nesterov_accelerated_gradient():
    ...
