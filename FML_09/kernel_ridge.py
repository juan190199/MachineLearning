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


