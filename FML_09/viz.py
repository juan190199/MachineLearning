import matplotlib.pyplot as plt

from utils import refine_fit


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


def show_refined_circles(models):
    circle_handles = []
    for method, color in [('algebraic', 'blue'), ('lm', 'orange')]:

        fit_models = refine_fit(models, method)
        for cr in fit_models:
            circle = plt.Circle(cr[:2], radius=cr[2], fill=False, edgecolor=color)
            plt.gca().add_patch(circle)
        # Remember the last circle for the legend
        circle_handles.append(circle)

    plt.legend(handles=circle_handles, labels=['Algebraic distance', 'Levenberg-Marquardt'])
