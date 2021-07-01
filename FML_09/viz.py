import matplotlib.pyplot as plt


def plot_circles_and_data(model, outliers):
    """

    :param model:
    :param outliers:
    :return:
    """
    plt.figure(figsize=(10, 10))
    for circ in model:
        plt.scatter(circ['MCS'][:, 0], circ['MCS'][:, 1], c='green')
        plt.gca().add_patch(plt.Circle(circ['c'], radius=circ['r'], fill=False))

    plt.scatter(outliers[:, 0], outliers[:, 1], c='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

