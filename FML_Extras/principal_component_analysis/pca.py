import numpy as np
import pandas as pd

from sklearn import base
from sklearn import preprocessing
from sklearn import utils

from . import svd


class PCA(base.BaseEstimator, base.TransformerMixin):
    """
    Principal Component Analysis (PCA)


    """

    def __init__(self, rescale_with_mean=True, rescale_with_std=True, n_components=2, n_iter=3,
                 copy=True, check_input=True, random_state=None, engine='auto', as_array=False):
        self.n_components = n_components
        self.n_iter = n_iter
        self.rescale_with_mean = rescale_with_mean
        self.rescale_with_std = rescale_with_std
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine
        self.as_array = as_array

    def fit(self, X, y=None):
        """

        :param X:
        :param y:
        :return:
        """
        # Check input
        if self.check_input:
            utils.check_array(X)

        # Convert pandas DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.tonumpy(dtype=np.float64)

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        self.n_features_in_ = X.shape[1]

        # Scale data
        if self.rescale_with_mean or self.rescale_with_std:
            self.scaler_ = preprocessing.StandardScaler(
                copy=False,
                with_mean=self.rescale_with_mean,
                with_std=self.rescale_with_std
            ).fit(X)
            X = self.scaler_.transform(X)

        # Compute SVD
        self.U_, self.s_, self.V_ = svd.compute_svd(
            X=X,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )

        # Compute total inertia
        self.total_inertia_ = np.sum(np.square(X)) / len(X)

        return self

    def check_is_fitted(self):
        utils.validation.check_is_fitted(self, 'total_inertia_')

    def transform(self, X):
        """

        :param X:
        :return:
        """
        self._check_is_fitted()
        if self.check_input:
            utils.check_array(X)
        rc = self.row_coordinates(X)
        if self.as_array:
            return rc.to_numpy()
        return rc

    def row_coordinates(self, X):
        """
        Returns the row principal coordinates
        :param X:
        :return:
        """
        self.check_is_fitted()

        # Extract index
        index = X.index if isinstance(X, pd.DataFrame) else None

        # Copy data
        if self.copy:
            X = np.array(X, copy=True)

        # Scale data
        if hasattr(self, 'scaler_'):
            X = self.scaler_.transform(X)

        return pd.DataFrame(data=X.dot(self.V_.T), index=index, dtype=np.float64)


