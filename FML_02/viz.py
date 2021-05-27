from matplotlib import pyplot as plt


def plot_data(X, y):
    """
    Plot 10 data images
    :param X: ndarray (n, 8, 8) -- Design matrix of 8x8 images
    :param y: ndarray (n, ) -- Label array
    :return:
    """
    _, axes = plt.subplots(2, 5)
    for ax, image, label in zip(axes[0, :], X[:5], y[:5]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)
    for ax, image, label in zip(axes[1, :], X[5:], y[5:]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='bicubic')
        ax.set_title('Training: %i' % label)
    plt.show


def plot_error_rate(K, confusion):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].set_title("Average error rate over all class pairs")
    axes[0].plot(K, confusion.mean(axis=2).mean(axis=1))
    axes[0].scatter(K, confusion.mean(axis=2).mean(axis=1), c="Red")
    axes[0].set_ylabel("average error in %")
    axes[0].set_xlabel("number of neighbors k")

    axes[1].set_title("Average error rate over all class pairs")
    axes[1].plot(K, confusion[:, 3, 9])
    axes[1].scatter(K, confusion[:, 3, 9], c="Red")
    axes[1].set_ylabel("average error in %")
    axes[1].set_xlabel("number of neighbors k")

    fig.tight_layout()
    plt.show()
