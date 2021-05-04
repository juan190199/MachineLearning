import numpy as np

from density_tree import DensityTree


class GenerativeClassifier:
    """
    Combines 10 DensityTree objects.
    One object of DensityTree is to be trained with only one class of data from the training data.

    To train the generative classifier with the full data set, it is needed to separate the data into 10 subsets
    and then train 10 DensityTree objects, each trained for a different subset.

    For prediction, all 10 DensityTree returns the prediction p(x|y) * p(y) for a data point.
    The target for the DensityTree maximizing p(x|y) * p(y) is the prediction of the generative classifier
    """
    def __init__(self):
        self.trees_ = [DensityTree() for i in range(10)]

    def fit(self, data, target, n_min=20):
        """

        :param data:
        :param target:
        :param n_min:
        :return:
        """
        data_subsets = [data[target == i] for i in range(10)]
        N = len(target)
        for i, tree in enumerate(self.trees_):
            tree.fit(data_subsets[i], len(data_subsets[i]) / N, n_min)

    def predict(self, x):
        """

        :param x: array-like of shape (1, n_features)
        :return:
        """
        return np.argmax([tree.predict(x) for tree in self.trees_])
