import logging
from typing import Callable

import numpy as np
from numpy.linalg import pinv

logger = logging.getLogger(__name__)


class GaussNewton:
    """
    Gauss-Newton:
    """
    def __init__(self):
        ...

    def fit(self):
        ...

    def predict(self):
        ...

    def get_residual(self):
        ...

    def get_estimate(self):
        ...

    def _calculate_residual(self):
        ...

    def _calculate_jacobian(self):
        ...

    @staticmethod
    def _calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
        ...
