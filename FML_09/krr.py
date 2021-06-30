import numpy as np
import scipy as sp
import scipy.sparse.linalg

from numpy.linalg import inv
from scipy import linalg


class KernelRidge:
    """

    """

    def __init__(self, kernel_type='linear', C=1.0, gamma=5.0):
        """

        :param kernel_type:
        :param C:
        :param gamma:
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
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1, x2.T) ** 2)

    def kernel_gaussian(self, x1, x2):
        """

        :param x1:
        :param x2:
        :return:
        """
        return np.exp(-linalg.norm(x1 - x2) ** 2 / (2 * (self.gamma ** 2)))

    def sparse_kernel_gaussian(self, x1, x2):
        """

        :param self:
        :param x1: ndarray of shape (dimension, )

        :param x2: ndarray of shape (n, dimension)

        :return:
        """
        sq_dist = np.sum((x1 - x2) ** 2, axis=-1, dtype=float)

        # Cutoff
        sq_dist[sq_dist > 30 * self.gamma ** 2] = np.inf

        return np.exp(-sq_dist / (2 * (self.gamma ** 2)))

    def compute_sparse_kernel_matrix(self, X1, X2):
        """

        :param self:
        :param X1:
        :param X2:
        :return:
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        # Gram matrix
        sparse_K = sp.sparse.dok_matrix((n1, n2))

        idx1 = np.arange(n1)
        idx_2 = np.arange(n2)
        for i in idx1:
            k = self.kernel(X1[i], X2)
            sparse_K[i, idx_2[k != 0]] = k[k != 0]

        return sparse_K.tocsc()

    def compute_kernel_matrix(self, X1, X2):
        """

        :param X1:
        :param X2:
        :return:
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        # Gram matrix
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])

        return K

    def fit(self, X, y, sparse=False):
        """

        :param self:
        :param X:
        :param y:
        :return:
        """
        if sparse is False:
            K = self.compute_kernel_matrix(X, X)
            self.alphas = sp.dot(inv(K + self.C * np.eye(np.shape(K)[0])), y.transpose())
        else:
            n = X.shape[0]
            idxs = np.arange(n)
            G = self.compute_sparse_kernel_matrix(X, X)

            G[idxs, idxs] += self.C
            self.alphas = sp.sparse.linalg.spsolve(G, y)

        return self

    def predict(self, X_train, X_test, sparse=False):
        """

        :param X_train:
        :param X_test:
        :return:
        """
        if sparse is False:
            k = self.compute_kernel_matrix(X_test, X_train)
            y_test = sp.dot(k, self.alphas).transpose()
        else:
            kernel_vects = self.compute_sparse_kernel_matrix(X_test, X_train)
            y_test = kernel_vects.dot(self.alphas)

        return y_test
