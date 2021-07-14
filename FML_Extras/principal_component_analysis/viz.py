import numpy as np
import matplotlib.pyplot as plt


def plot_figure(X1, Y, X2=None, figsize=(8, 4)):
    """

    :param X:
    :param y:
    :return:
    """
    if X2 is not None:
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.title('My PCA')
        plt.scatter(X1[:, 0], X1[:, 1], c=Y)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        plt.figure(figsize=figsize)
        plt.subplot(122)
        plt.title('sklearn PCA')
        plt.scatter(X2[:, 0], X2[:, 1], c=Y)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=figsize)
        plt.title("My PCA")
        plt.scatter(X1[:, 0], X1[:, 1], c=Y)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
