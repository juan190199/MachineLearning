import numpy as np

from decision_tree import DecisionTree


class DecisionForest():
    """

    """

    def __init__(self, n_trees):
        self.trees_ = [DecisionTree() for i in range(n_trees)]

    def fit(self, data, labels, n_min=0):
        """

        :param data:
        :param labels:
        :param n_min:
        :return:
        """
        for tree in self.trees_:
            bootstrap = np.random.choice(len(data), size=len(data))
            tree.fit(data[bootstrap], labels[bootstrap], n_min)

    def predict(self, x):
        """

        :param x:
        :return:
        """
        return np.mean([tree.predict(x) for tree in self.trees_])
