import numpy as np

from density_tree import DensityTree


class DensityForest():
    """

    """

    def __init__(self, n_trees):
        # create ensemble
        self.trees_ = [DensityTree() for i in range(n_trees)]

    def fit(self, data, prior, n_min=20):
        """

        :param data:
        :param prior:
        :param n_min:
        :return:
        """
        for tree in self.trees_:
            bootstrap_data = data[np.random.choice(len(data), size=len(data))]
            tree.fit(bootstrap_data, prior, n_min)

    def predict(self, x):
        """

        :param x:
        :return:
        """
        return np.mean([tree.predict(x) for tree in self.trees_])
