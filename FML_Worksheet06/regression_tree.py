import numpy as np


class Node:
    pass


class Tree:
    def __init__(self):
        self.root = Node()

    def find_leaf(self, x):
        node = self.root
        while hasattr(node, "feature"):
            j = node.feature
            if x[j] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node


def make_regression_split_node(node, feature_indices):
    """

    :param node:
    :param feature_indices:
    :return:
    """
    pass


def make_regression_leaf_node(node):
    """

    :param node:
    :return:
    """
    pass


class RegressionTree(Tree):
    def __init__(self):
        super(RegressionTree, self).__init__()

    def fit(self, data, labels, n_min=200):
        """

        :param data:
        :param labels:
        :param n_min:
        :return:
        """
        N, D = data.shape
        D_try = int(np.sqrt(D))

        # Initialize root node
        self.root.data = data
        self.root.labels = labels

        # Build the tree
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0]
            if n >= n_min:
                perm = np.random.permutation(D)
                left, right = make_regression_split_node(node, perm[:D_try])
                stack.append(left)
                stack.append(right)
            else:
                make_regression_leaf_node(node)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        if X.shape[0] == 1:
            leaf = self.find_leaf(X)
            return leaf.response
        else:
            pred = np.apply_along_axis(self.predict, axis=1, arr=X)
