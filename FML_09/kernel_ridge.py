import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse.linalg
from numpy.linalg import inv
from scipy import linalg

# https://github.com/supenova1604/kernel_ridge_regr/blob/master/kerner_ridge_regr.py
# https://github.com/ptocca/KRRPM/blob/main/src/krrpm/krrpm.py

class KernelRidge:
    """

    """

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
            'gaussian': self.kernel_gaussian,
            's_gaussian': self.sparse_kernel_gaussian
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

    def sparse_kernel_gaussian(self, x1, x2, gamma=5.0):
        gamma = self.gamma
        norm = np.sum((x1 - x2) ** 2, axis=-1, dtype=float)

        # Cutoff
        norm[norm > 30 * (gamma ** 2)] = 0

        return np.exp(-(norm ** 2) / (2 * (gamma ** 2)))

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

    def compute_sparse_kernel_matrix(self, X1, X2):
        """

        :param X1:
        :param X2:
        :return:
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        sparse_K = sp.sparse.dok_matrix((n1, n2))
        idx1 = np.arange(n1)
        idx2 = np.arange(n2)

        for i in idx1:
            k = self.kernel(X1[i], X2)
            sparse_K[i, idx2[k != 0]] = k[k != 0]

        return sparse_K.tocsc()

    def fit(self, X, y, sparse=False):
        """
        Training kernel ridge regression

        :param X: np.ndarray of shape (n, d)
            Design matrix with n d-dimensional instances

        :param y: np.ndarray of shape (n, 1)
            output vector for design matrix

        :return:
        """
        if sparse is False:
            K = self.compute_kernel_matrix(X, X)
            self.alphas = sp.dot(inv(K + self.C * np.eye(np.shape(K)[0])),
                                 y.transpose())
        else:
            sparse_K = self.compute_sparse_kernel_matrix(X, X)
            self.alphas = scipy.sparse.linalg.spsolve(sparse_K + self.C * np.eye(np.shape(sparse_K)[0]), y)

        return self.alphas

    def predict(self, X_train, X_test, sparse=False):
        """

        :param X_train:
        :param X_test:
        :return:
        """
        if sparse is False:
            K = self.compute_kernel_matrix(X_test, X_train)
        else:
            K = self.compute_sparse_kernel_matrix(X_test, X_train)

        y_test = sp.dot(K, self.alphas)
        return y_test.transpose
