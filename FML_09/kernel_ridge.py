import numpy as np
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

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
