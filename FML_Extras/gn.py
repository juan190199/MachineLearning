import logging
from typing import Callable

import numpy as np
from numpy.linalg import pinv

logger = logging.getLogger(__name__)


class GaussNewton:
    """
    Gaussian-Newton: Given output vector y, design matrix X and model f, minimize the squared sum of residuals
    """

    def __init__(self,
                 hypothesis: Callable,
                 max_iter: int = 1000,
                 tolerance_difference: float = 1e-16,
                 tolerance: float = 1e-9,
                 init_guess: np.ndarray = None):
        """

        :param hypothesis: hypothesis function to be fitted
        :param max_iter:
        :param tolerance_difference:
        :param tolerance:
        :param init_guess:
        """
        self.hypothesis_ = hypothesis
        self.max_iter_ = max_iter
        self.tolerance_difference_ = tolerance_difference
        self.tolerance_ = tolerance
        self.coefficients_ = None
        self.X = None
        self.y = None
        self.init_guess_ = None
        if init_guess is None:
            self.init_guess_ = init_guess

    def fit(self, X: np.ndarray, y: np.ndarray, init_guess: np.ndarray = None) -> np.ndarray:
        """
        Fit coefficients by minimizing RMSE
        :param X:
        :param y:
        :param init_guess:
        :return:
        """
        self.X = X
        self.y = y
        if init_guess is not None:
            self.init_guess_ = init_guess
        if init_guess is None:
            raise Exception("Initial guess needs to be provided")

        self.coefficients_ = self.init_guess_
        rmse_prev = np.inf
        for k in range(self.max_iter_):
            residual = self.get_residual()
            jacobian = self._calculate_jacobian(self.coefficients_, step=1e-6)
            self.coefficients_ = self.coefficients_ - np.dot(self._calculate_pseudoinverse(jacobian), residual)
            rmse = np.sqrt(np.sum(residual ** 2))
            logger.info(f"Round {k}: RMSE {rmse}")

            if self.tolerance_difference_ is not None:
                diff = np.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference_:
                    logger.info("RMSE difference between iterations smaller than tolerance. Fit terminated.")
                    return self.coefficients_

            if rmse < self.tolerance_:
                logger.info("RMSE error smaller than tolerance. Fit terminated.")
                return self.coefficients_

            rmse_prev = rmse

        logger.info("Max. number of iterations reached. Fit didn't converge.")
        return self.coefficients_

    def predict(self, x: np.ndarray):
        """

        :param x:
        :return:
        """
        ...

    def get_residual(self) -> np.ndarray:
        """

        :return:
        """
        ...

    def get_estimate(self) -> np.ndarray:
        """

        :return:
        """
        ...

    def _calculate_residual(self, coefficients: np.ndarray) -> np.ndarray:
        """

        :param coefficients:
        :return:
        """
        ...

    def _calculate_jacobian(self, x0: np.ndarray, step: float = 1e-6) -> np.ndarray:
        """

        :param x0:
        :param step:
        :return:
        """
        ...

    @staticmethod
    def _calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
        """

        :param x:
        :return:
        """
        ...
