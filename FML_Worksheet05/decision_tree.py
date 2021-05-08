import numpy as np

from base import (Node, Tree)


def make_decision_split_node(node, feature_indices):
    """

    :param node: Node
        Node to be split

    :param feature_indices: array-like of shape (D_try, )
        Contains feature indices to be considered in the present split

    :return: tuple
        Tuple of left and right children nodes (to be placed on the stack)
    """
    n, D = node.data.shape

    # Find best feature j (among 'feature_indices') and best threshold t for the split
    e_min = 1e100
    j_min, t_min = 0, 0
    for j in feature_indices:
        # Remove duplicate features
        dj = np.sort(np.unique(node.data[:, j]))
        # Compute candidate thresholds
        tj = (dj[1:] + dj[:-1]) / 2

        # Compute Gini-impurity of resulting children nodes for each candidate threshold
        for t in tj:
            left_indices = node.data[:, j] <= t

            nl = np.sum(node.data[:, j] <= t)
            ll = node.labels[left_indices]
            el = nl * (1 - np.sum(np.square(np.bincount(ll) / nl)))

            nr = n - nl
            # lr = node.labels[node.data[:, j] > t]
            lr = node.labels[~left_indices]
            er = nr * (1 - np.sum(np.square(np.bincount(lr) / nr)))

            if el + er < e_min:
                e_min = el + er
                j_min = j
                t_min = t

    # Create children
    left = Node()
    right = Node()

    # Initialize 'left' and 'right' with the data subsets and labels
    # according to the optimal split found above
    left.data = node.data[node.data[:, j_min] <= t_min, :]
    left.labels = node.labels[node.data[:, j_min] <= t_min]

    right.data = node.data[node.data[:, j_min] > t_min, :]
    right.labels = node.labels[node.data[:, j_min] > t_min]

    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min

    return left, right


def make_decision_leaf_node(node):
    """
    Compute and store leaf response

    :param node: Node
        Node that has reached termination criterion

    :return:
    """
    node.N = node.labels.shape[0]
    node.response = np.bincount(node.labels, minlength=10) / node.N


def node_is_pure(node):
    """
    Check if node contains instances of the same class

    :param node: Node
        Node to be checked if has reached termination criterion

    :return: bool
    True, if given node is pure. Otherwise, false
    """
    return np.unique(node.labels).shape[0] == 1


class DecisionTree(Tree):
    """
    Decision Tree: Tree-like model that predicts the value of a target variable based on several input variables.
    """

    def __init__(self):
        super(DecisionTree, self).__init__()

    def fit(self, data, labels, n_min=20):
        """

        :param data: array-like of shape (n_instances, n_features)
            Design matrix

        :param labels: array-like of shape (n_instances, )
            Ground-truth responses

        :param n_min: int, default=20
            The minimum number of samples required to split an internal node

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
        """
        Computes p(y | x)

        :param x: array-like of shape (1, n_features)
            Sample point (test instance)

        :return: int
            Returns posterior probability p(y | x)
        """
        leaf = self.find_leaf(x)
        return leaf.response
