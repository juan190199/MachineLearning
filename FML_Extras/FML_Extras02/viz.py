import numpy as np
import matplotlib.pyplot as plt


def plot_iris_ds(X1, Y, X2=None, figsize=(8, 4)):
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


def plot_digits_ds(X1, y, X2=None, figsize=(8, 4)):
    if X2 is not None:
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.title('My PCA')
        plt.scatter(X1[:, 0], X1[:, 1],
                    c=y, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.tight_layout()

        plt.figure(figsize=figsize)
        plt.subplot(122)
        plt.title('sklearn PCA')
        plt.scatter(X2[:, 0], X2[:, 1],
                    c=y, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.colorbar()
        plt.show()

    else:
        plt.scatter(X1[:, 0], X1[:, 1],
                    c=y, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar()

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
