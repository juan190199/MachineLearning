import numpy as np

from base import Tree


class DecisionTree(Tree):
    def __init__(self):
        super(DecisionTree, self).__init__()

    def train(self, data, labels, n_min=20):
        '''
        data: the feature matrix for all digits
        labels: the corresponding ground-truth responses
        n_min: termination criterion (don't split if a node contains fewer instances)
        '''
        N, D = data.shape
        D_try = int(np.sqrt(D)) # how many features to consider for each split decision

        # initialize the root node
        self.root.data = data
        self.root.labels = labels

        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0] # number of instances in present node
            if n >= n_min and not node_is_pure(node):
                # Call 'make_decision_split_node()' with 'D_try' randomly selected
                # feature indices. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                ... # your code here
            else:
                # Call 'make_decision_leaf_node()' to turn 'node' into a leaf node.
                ... # your code here

    def predict(self, x):
        leaf = self.find_leaf(x)
        # compute p(y | x)
        return ... # your code here
