import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data(X_train, X_train_std, y_train):
    """

    :param models:
    :param X:
    :param y:
    :param ellipse:
    :return:
    """
    targets = np.unique(y_train)

    fig, axs = plt.subplots(nrows=len(targets), ncols=2, figsize=(10, 10))
    for i in range(len(targets)):
        axs[i, 0].imshow(np.array(np.mean(X_train[y_train==i], axis=0)).reshape((8, 8)), cmap='gray')
        fig.colorbar(cm.ScalarMappable(cmap='gray'), ax=axs[i, 0])
        axs[i, 0].set_xticks(())
        axs[i, 0].set_yticks(())

        axs[i, 1].imshow(np.array(np.mean(X_train_std[y_train==i], axis=0)).reshape((8, 8)))
        fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=axs[i, 1])
        axs[i, 1].set_xticks(())
        axs[i, 1].set_yticks(())

        axs[i, 0].set_title("Mean image with target {}".format(i))
        axs[i, 1].set_title("Standardized mean of image of target {}".format(i))

    plt.tight_layout()
    plt.show()
