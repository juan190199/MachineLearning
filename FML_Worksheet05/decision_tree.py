import numpy as np

from base import Tree


def make_decision_split_node(node, feature_indices):
    """

    :param node:
    :param feature_indices:
    :return:
    """
    pass


def make_decision_leaf_node(node):
    """

    :param node:
    :return:
    """
    pass


def node_is_pure(node):
    """
    Check if node contains instances of the same class
    :param node:
    :return: bool
    True, if given node is pure. Otherwise, false
    """
    return np.unique(node.labels).shape[0] == 1


class DecisionTree(Tree):
    """
    Decision Tree:

    Attributes
    -----------

    """
    def __init__(self):
        super(DecisionTree, self).__init__()

    def train(self, data, labels, n_min=20):
        """

        :param data: array-like of shape (n_instances, n_features)
            Design matrix

        :param labels: array-like of shape (n_instances, )
            Ground-truth responses

        :param n_min: int, default=20
            Termination criterion (Do not split if node contains fewer instances)

        :return:
        """
        N, D = data.shape
        D_try = int(np.sqrt(D))  # Number of features to consider for each split decision

        # Initialize the root node
        self.root.data = data
        self.root.labels = labels

        # Build the tree
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0]  # Number of instances in present node
            if n >= n_min and not node_is_pure(node):
                perm = np.random.permutation(D)
                left, right = make_decision_split_node(node, perm[:D_try])
                stack.append(left)
                stack.append(right)
            else:
                make_decision_leaf_node(node)

    def predict(self, x):
        leaf = self.find_leaf(x)
        # compute p(y | x)
        return ...  # your code here
