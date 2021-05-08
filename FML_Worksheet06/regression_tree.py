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
    n, D = node.data.shape
    # Find best feature j (among 'feature_indices') and best threshold t for the split
    e_min = 1e100
    j_min, t_min = 0, 0
    for j in feature_indices:
        # Duplicate feature values have to be removed because candidate thresholds are
        # the midpoints of consecutive feature values, not the feature value itself
        dj = np.sort(np.unique(node.data[:, j]))
        # Compute candidate thresholds
        tj = (dj[1:] + dj[:-1]) / 2

        # Compute Gini-impurity of resulting children nodes for each candidate threshold
        for t in tj:
            left_indices = node.data[:, j] <= t
            nl = np.sum(left_indices)
            ll = node.labels[left_indices]
            # el = nl * (1 - np.sum(np.square(np.bincount(ll) / nl)))
            el = ((ll - ll.mean()) ** 2).sum()
            nr = n - nl
            # lr = node.labels[node.data[:, j] > t]
            lr = node.labels[~left_indices]
            # er = nr * (1 - np.sum(np.square(np.bincount(lr) / nr)))
            er = ((lr - lr.mean())**2).sum()

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
