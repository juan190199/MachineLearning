from matplotlib import pyplot as plt
import matplotlib as mpl

import numpy as np
from scipy import linalg

from nearest_mean_classifier import nearest_mean_classifier


def scatter_plot_data(X_train, X_test, y_train, y_test):
    """
    Scatter plot of 2d data
    :param X_train: ndarray (n, 2) -- Design matrix
    :param X_test: ndarray (n, ) -- Outcome-label vector for training data
    :param y_train: ndarray (m, 2) -- Test data matrix
    :param y_test: ndarray (m, ) -- Outcome-label vector for test data
    :return:
    """
    # fig = plt.figure(figsize=(10, 3))

    plt.subplot(121)
    plt.title('Training data')
    for i, label in enumerate(np.unique(y_train)):
        plt.scatter(X_train[y_train == label, 0],
                    X_train[y_train == label, 1],
                    cmap='Paired',
                    label=f'Label {label}')
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(122)
    plt.title('Test data')
    for i, label in enumerate(np.unique(y_test)):
        plt.scatter(X_test[y_test == label, 0],
                    X_test[y_test == label, 1],
                    cmap='Paired',
                    label=f'Label {label}')
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('Feature 1')

    plt.tight_layout()
    plt.show()


def plot_data_nm_classifier(X_train, X_test, y_train, y_test):
    """
    Plot of decision region for nearest mean classifier
    :param X_train: ndarray (n, 2) -- Design matrix
    :param X_test: ndarray (n, ) -- Outcome-label vector for training data
    :param y_train: ndarray (m, 2) -- Test data matrix
    :param y_test: ndarray (m, ) -- Outcome-label vector for test data
    :return:
    """
    # Define bounds of the domain
    feat_min, feat_max = np.min(X_test, axis=0), np.max(X_test, axis=0)
    x, y = np.linspace(feat_min[0], feat_max[0], 10000), np.linspace(feat_min[1], feat_max[1], 10000)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    xx, yy = xv.flatten(), yv.flatten()
    xx, yy = xv.reshape((len(xx), 1)), yv.reshape((len(yy), 1))

    grid = np.hstack((xx, yy))

    # Make predictions for the grid
    grid_prediction, means = nearest_mean_classifier(X_train, grid, y_train)

    # Reshape the predictions back into a grid
    zz = grid_prediction.reshape(xv.shape)

    plt.contourf(xv, yv, zz, cmap='Paired')

    # Scatter data
    labels = np.unique(y_test)
    for i, label in enumerate(labels):
        plt.scatter(X_test[y_test == label, 0],
                    X_test[y_test == label, 1],
                    cmap='Paired',
                    label=f'Label {label}')

    for i, label in zip(np.arange(means.shape[0]), labels):
        plt.scatter(means[i][0], means[i][1], marker='X', s=30, color='#2ca02c', label=f'Mean label {label}')

    plt.xlim(feat_min[0], feat_max[0])
    plt.ylim(feat_min[1], feat_max[1])

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()


def plot_data(models, X_test, y_test, ellipse=False):
    """

    :param models:
    :param X:
    :param y:
    :param ellipse:
    :return:
    """
    classes = np.unique(y_test)
    # Number of models
    n_models = len(models)

    # Define bounds of the domain
    feat_min, feat_max = np.min(X_test, axis=0), np.max(X_test, axis=0)
    x, y = np.linspace(feat_min[0], feat_max[0], 10000), np.linspace(feat_min[1], feat_max[1], 10000)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    xx, yy = xv.flatten(), yv.flatten()
    xx, yy = xv.reshape((len(xx), 1)), yv.reshape((len(yy), 1))

    grid = np.hstack((xx, yy))

    fig, ax = plt.subplots(1, n_models, figsize=(15, 5))
    for i in range(n_models):
        # Make predictions for the grid
        grid_pred = models[i].predict(grid)
        # Reshape the predictions back into a grid
        zz = grid_pred.reshape(xv.shape)
        ax[i].contourf(xv, yv, zz, cmap='Paired')

        # Scatter plot data
        for j, group in enumerate(classes):
            ax[i].scatter(X_test[y_test == group, 0],
                          X_test[y_test == group, 1],
                          cmap='Paired',
                          label=f'Label {group}')
            ax[i].legend()

        for mean in models[i].means_:
            ax[i].scatter(mean[0], mean[1], marker='X', s=30, color='#2ca02c')

            ax[i].set_xlim(feat_min[0], feat_max[0])
            ax[i].set_ylim(feat_min[1], feat_max[1])

            ax[i].set_xticks(())
            ax[i].set_yticks(())

            ax[i].set_xlabel("Feature 1")
            ax[i].legend()

            ax[0].set_ylabel("Feature 2")

            plt.tight_layout()
