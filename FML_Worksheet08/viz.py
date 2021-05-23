import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data(X_train, X_train_std, y_train, filters):
    """

    :param X_train:
    :param X_train_std:
    :param y_train:
    :param filters:
    :return:
    """
    targets = np.unique(y_train)

    fig, axs = plt.subplots(nrows=len(targets), ncols=2, figsize=(10, 10))
    for i in range(len(targets)):
        axs[i, 0].imshow(np.array(np.mean(X_train[y_train == i], axis=0)).reshape((8, 8)), cmap='gray')
        fig.colorbar(cm.ScalarMappable(cmap='gray'), ax=axs[i, 0])
        axs[i, 0].set_xticks(())
        axs[i, 0].set_yticks(())

        axs[i, 1].imshow(np.array(np.mean(X_train_std[y_train == i], axis=0)).reshape((8, 8)))
        fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=axs[i, 1])
        axs[i, 1].set_xticks(())
        axs[i, 1].set_yticks(())

        axs[i, 0].set_title("Mean image with label {}".format(filters[i]))
        axs[i, 1].set_title("Standardized mean of image of label {}".format(filters[i]))

    plt.tight_layout()
    plt.show()


def plot_theta_omp(theta, theta_std, filters):
    """

    :param theta:
    :return:
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.ylabel('Feature')
    plt.title(r'$\theta$ for OMP classification of {} & {}'.format(filters[0], filters[1]))
    plt.imshow(theta)
    plt.colorbar()

    plt.subplot(122)
    plt.ylabel('Feature')
    plt.imshow(theta_std)
    plt.title(r'$\theta$ for OMP classification of 1 & 7 (standardized)')
    plt.colorbar()
    plt.show()


def plot_acc(acc, acc_std):
    plt.plot(acc[0, :], 'r', linestyle='--', label='training acc NOT std')
    plt.plot(acc[1, :], 'r', label='testing acc NOT std')
    plt.plot(acc_std[0, :], 'b', linestyle='--', label='training acc std')
    plt.plot(acc_std[1, :], 'b', label='testing acc std')
    plt.ylabel('Accuracy')
    plt.xlabel('t-1')
    plt.legend()
    plt.tight_layout()
    plt.show()
