import numpy as np
import scipy as sp
from numpy.linalg import inv
from scipy import linalg


# https://github.com/supenova1604/kernel_ridge_regr/blob/master/kerner_ridge_regr.py
# https://github.com/ptocca/KRRPM/blob/main/src/krrpm/krrpm.py

class KernelRidge():
    """"""

    def __init__(self, kernel_type='linear', C=1.0, gamma=5.0):
        """

        :param kernel_type: string, default='linear'
            Kernel type to use in training. Options are:
            'linear': uses linear kernel function
            'quadratic': uses quadratic kernel function
            'gaussian': uses gaussian kernel function

        :param C: float, default=1.0
            Value of regularization parameter

        :param gamma: float, default=5.0
            Parameter for gaussian kernel or polynomial kernel
        """
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic,
            'gaussian': self.kernel_gaussian
        }
        self.kernel_type = kernel_type
        self.kernel = self.kernels[self.kernel_type]
        self.C = C
        self.gamma = gamma

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1.T, x2)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1.T, x2) ** 2)

    def kernel_gaussian(self, x1, x2, gamma=5.0):
        gamma = self.gamma
        return np.exp(-linalg.norm(x1 - x2) ** 2 / (2 * (gamma ** 2)))

    def compute_kernel_matrix(self, X1, X2):
        """
        Compute Gram matrix given two input matrices

        :param X1: np.ndarray of shape ()

        :param X2: np.ndarray of shape ()

        :return: np.ndarray of shape (X1.shape[0], X2.shape[0])
            Kernel matrix (Gram matrix)
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        # Gram matrix
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])

        return K

    def fit(self, X, y):
        """
        Training kernel ridge regression

        :param X: np.ndarray of shape (n, d)
            Design matrix with n d-dimensional instances

        :param y: np.ndarray of shape (n, 1)
            output vector for design matrix

        :return:
        """
        K = self.compute_kernel_matrix(X, X)
        self.alphas = sp.dot(inv(K + self.C * np.eye(np.shape(K)[0])),
                            y.transpose())

        return self.alphas

    def predict(self, X_train, X_test):
        """

        :param X_train:
        :param X_test:
        :return:
        """
        K = self.compute_kernel_matrix(X_test, X_train)

        y_test = sp.dot(K, self.alphas)
        return y_test.transpose
