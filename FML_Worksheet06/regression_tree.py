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
        pass
