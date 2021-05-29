import logging
from typing import Callable

import numpy as np
from numpy.linalg import pinv


class GaussianNewton:
    """
    Gaussian-Newton: Method for solving non-linear least squares problems.
    Minimize a sum of squared function values without requiring to compute second derivatives

    Attributes
    ----------
    """

    def __init__(self,
                 hypothesis,
                 max_iter=1000,
                 tol_dif=1e-16,
                 tol=1e-9,
                 init_guess=None):
        """

        :param hypothesis: Callable
            Hypothesis function

        :param max_iter: int, default=1000
            Maximum number of iterations

        :param tol_dif: float, default=1e-16
            Tolerance for difference between previous least squares and actual least squares

        :param tol: float, default=1e-9
            Tolerance for least squares

        :param init_guess: list of length: n_parameters
            Initial guess for parameters of hypothesis model
        """
        self.hypothesis_ = hypothesis
        self.max_iter_ = max_iter
        self.tol_dif_ = tol_dif
        self.tol_ = tol
        self.init_guess = init_guess
        self.parameters = None
        self.X = None
        self.y = None

    def fit(self, X, y, init_guess):
        """

        :param X:
        :param y:
        :param init_guess:
        :return:
        """
