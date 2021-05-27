import numpy as np
from matplotlib import pyplot as plt


def plot_gen_data(filt, means, covariances):
    """

    :param filt:
    :return:
    """
    if filt is None:
        filt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        n_rows = len(filt)
        fig, ax = plt.subplots(n_rows, 5, figsize=(20, 30))
    else:
        n_rows = len(filt)
        fig, ax = plt.subplots(n_rows, 5, figsize=(15, 5))

    samples = np.empty((len(filt), 5, 64))
    # Generate samples
    for i in range(5):
        for k in range(len(filt)):
            samples[k, i, :] = np.random.multivariate_normal(mean=means[k], cov=covariances[k])
            ax[k, i].imshow(samples[k, i, :].reshape((8, 8)), cmap='gray')
            ax[k, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[k, i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    plt.tight_layout()
    plt.show()
