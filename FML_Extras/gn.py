import logging
from typing import Callable

import numpy as np
from numpy.linalg import pinv

logger = logging.getLogger(__name__)


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
                 conv=1e-6,
                 tol_dif=1e-16,
                 tol=1e-9):
        """

        :param hypothesis: Callable
            Hypothesis function

        :param max_iter: int, default=1000
            Maximum number of iterations

        :param conv: float, default=1e-6
            Convergence criteria. np.abs((lse_prev - lse) / lse_prev)

        :param tol_dif: float, default=1e-16
            Tolerance for difference between previous least squares and actual least squares

        :param tol: float, default=1e-9
            Tolerance for least squares

        :param init_guess: list of length: n_parameters
            Initial guess for parameters of hypothesis model
        """
        self.hypothesis_ = hypothesis
        self.max_iter_ = max_iter
        self.conv_ = conv
        self.tol_dif_ = tol_dif
        self.tol_ = tol
        self.parameters_ = None
        self.init_guess = None
        self.X = None
        self.y = None

    def fit(self, X, y, init_guess):
        """
        Fit parameters by minimizing least squares problem
        :param X: np.ndarray of shape ()

        :param y: np.ndarray of shape ()

        :param init_guess: list of length:

        :return: np.ndarray of shape (n_parameters, )
            Fitted coefficients
        """
        self.X = X
        self.y = y
        if init_guess is not None:
            self.init_guess_ = init_guess
        if init_guess is None:
            raise Exception("Initial guess needs to be provided")

        self.parameters_ = self.init_guess_
        lse_prev = np.inf

        for k in range(self.max_iter_):
            residual = self.get_residual()
            jacobian = self._calculate_jacobian(self.parameters_, step=1e-6)
            new_parameters = self.parameters_ - np.dot(self._calculate_pseudoinverse(jacobian).T, residual)
            lse = np.sum(residual ** 2)
            logger.info(f"Round {k}: LSE {lse}")

            if self.conv_ is not None:
                diff = np.abs((lse_prev - lse) / lse_prev)
                if diff < self.conv_:
                    logger.info("Convergence criteria reached. Fit terminated.")
                    self.parameters_ = new_parameters
                    return self.parameters_

            if self.tol_dif_ is not None:
                diff = np.abs(lse_prev - lse)
                if diff < self.tol_dif_:
                    logger.info("LSE difference between iterations smaller than tolerance. Fit terminated.")
                    self.parameters_ = new_parameters
                    return self.parameters_

            if lse < self.tol_:
                logger.info("LSE error smaller than tolerance. Fit terminated.")
                self.parameters_ = new_parameters
                return self.parameters_

            lse_prev = lse
            self.parameters_ = new_parameters

        logger.info("Max. number of iterations reached. Fit didn't converge.")
        return self.parameters_

    def predict(self, X):
        """

        :param X:
        :return:
        """
        return self.hypothesis_(X, self.parameters_)

    def get_residual(self):
        """

        :return:
        """
        return self._calculate_residual(self.parameters_)

    def get_estimate(self):
        """

        :return:
        """
        return self.hypothesis_(self.X, self.parameters_)

    def _calculate_residual(self, parameters):
        """

        :param parameters:
        :return:
        """
        y_est = self.hypothesis_(self.X, parameters)
        return self.y - y_est

    def _calculate_jacobian(self, theta0, step=1e-6):
        """
        Calculate Jacobian matrix numerically at x0
        :param theta0: np.ndarray of shape (n_dim, 1)


        :param step: float, default=1e-6
            Increment for numerical approximation

        :return:
        """
        jacobian = []
        for i, parameter in enumerate(theta0):
            l_theta = theta0.copy()
            u_theta = theta0.copy()
            l_theta[i] -= step
            u_theta[i] += step
            l_residual = self._calculate_residual(l_theta)
            u_residual = self._calculate_residual(u_theta)

            part_derivative = (u_residual - l_residual) / (2 * step)
            jacobian.append(part_derivative)

        jacobian = np.asarray(jacobian).squeeze().T

        return jacobian

    @staticmethod
    def _calculate_pseudoinverse(X):
        """

        :param X:
        :return:
        """
        return pinv(np.dot(np.dot(X.T, X), X.T))
