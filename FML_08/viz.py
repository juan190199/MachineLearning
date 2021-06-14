import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_data(X_train, X_train_std, y_train, filters):
    """
    Plot mean of non-standardized data and standardized data

    :param X_train: ndarray of shape (n_instances, n_features)
        Design matrix

    :param X_train_std: ndarray of shape (n_instances, n_features)
        Standardized design matrix

    :param y_train: ndarray of shape (n_instances, )
        Output vector for train data

    :param filters: list
        List with target filters

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
    Plot weight vector for non-standardized data and standardized data

    :param theta: ndarray of shape (n_features,)
        Weight vector

    :param theta_std:
        Weight vector for standardized data

    :param filters: list
        List with target filters

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
    """
    Plot accuracy for non-standardized and standardized data with the optimal weight vector for each iteration of OMP

    :param acc: ndarray of shape (2, n_iterations)
        Accuracy for train and test non-standardized data

    :param acc_std: ndarray of shape (2, n_iterations)
        Accuracy for train and test standardized data

    :return:
    """
    plt.plot(acc[0, :], 'r', linestyle='--', label='training acc NOT std')
    plt.plot(acc[1, :], 'r', label='testing acc NOT std')
    plt.plot(acc_std[0, :], 'b', linestyle='--', label='training acc std')
    plt.plot(acc_std[1, :], 'b', label='testing acc std')
    plt.ylabel('Accuracy')
    plt.xlabel('t-1')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pixel(theta, mean=np.ones(64)):
    """
    Plot the order of pixels/features added to the correct subset of OMP with its target
    :param theta: ndarray of shape (n_features, n_iterations)
        Optimal weight vector for each iteration of OMP

    :param mean:
    :return:
    """
    old_idx = []
    im = np.zeros((8, 8))
    im_v = np.zeros((8, 8))
    for j in range(theta.shape[1]):
        idx = np.where(theta[:, j] != 0)
        for i in idx[0]:
            if i not in old_idx:
                new = i
                iu = np.unravel_index(i, (8, 8))
                im[iu] = theta.shape[1] + 1 - j
                old_idx.append(i)
                if theta[i, j] * mean[i] > 0:
                    vote = 1
                    im_v[iu] = 1
                else:
                    vote = 7
                    im_v[iu] = 2

        plt.figure()
        plt.subplot(121)
        plt.axis('off')
        plt.title('Add pixel {} ≡ {}'.format(new, iu))
        plt.imshow(im, vmin=0, vmax=theta.shape[1] + 1, cmap='jet')
        plt.subplot(122)
        plt.axis('off')
        plt.title('Votes for {}'.format(vote))
        plt.imshow(im_v, vmin=0, vmax=2, cmap='jet')


def plot_theta_viz(theta_classes, T_one_vs_rest):
    plt.figure(figsize=(16, 10))
    plt.imshow(theta_classes, cmap='jet')
    plt.title(r'Visualization of $\theta$ with T = ' + str(T_one_vs_rest))
    plt.xlabel('Classes')
    plt.ylabel('Features')
    plt.colorbar()
    plt.show()


def plot_unknown_data(unknowns, X_test, y_test, y_predict):
    for u in unknowns:
        plt.figure()
        plt.imshow(X_test[u, :].reshape(8, 8), cmap='gray')
        plt.axis('off')
        plt.title('Test image {}, if not "unkn": prediction = {}, true label = {}'.format(u, y_predict[u], y_test[u]))
        plt.show()
