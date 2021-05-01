from matplotlib import pyplot as plt
import numpy as np

from classifiers import error_threshold_classifier


def plot_data(data):
    """
    Plot prior and likelihood for each class
    :param data: ndarray -- Data set to be used
    :return:
    """
    X_0 = data[data[:, 1] == 0][:, 0]
    X_1 = data[data[:, 1] == 1][:, 0]
    fig, ax = plt.subplots(1, 5, figsize=(10, 3))
    ax[0].scatter(data[:, 1], data[:, 0], alpha=0.3, color='black')
    ax[0].set_title('Prior for class $Y$')
    ax[0].set_xlabel("Classes")
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylabel("Data points")
    ax[0].set_ylim(-0.5, 1.5)

    ax[1].hist2d(data[:, 1], data[:, 0], bins=(2, 20))
    ax[1].set_title('Prior for class $Y$')
    ax[1].set_xlabel("Classes")

    ax[2].bar([0, 1], [X_0.size, X_1.size], width=0.6)
    ax[2].set_xticks([0, 1])
    ax[2].set_xlim([-0.5, 1.5])
    ax[2].set_yticks([0, 25000, 50000])
    ax[2].set_title(r'Prior for class $Y$')
    ax[2].set_xlabel("Classes")

    ax[3].hist(X_0, 50, density=True, facecolor='green', alpha=0.5)
    ax[3].set_ylabel(r'$p(X = x \mid Y = 0)$')
    ax[3].plot([0, 1], [2, 0])
    ax[3].set_title(r'Likelihood for $Y=0$')
    ax[3].set_xlabel("Classes")

    ax[4].hist(X_1, 50, density=True, facecolor='blue', alpha=0.5)
    ax[4].set_ylabel(r'$p(X = x \mid Y = 1)$')
    ax[4].plot([0, 1], [0, 2])
    ax[4].set_title(r'Likelihood for $Y=1$')
    ax[4].set_xlabel("Classes")

    plt.tight_layout()
    plt.show()


def plot_cdf(x_0, x_1):
    """
    Plot cumulative distribution function for each class
    :param x_0: ndarray -- Input data with label 0
    :param x_1: ndarray -- Input data with label 1
    :return:
    """
    fig = plt.figure(figsize=(10, 3))
    domain = np.linspace(0, 1, 50)

    plt.subplot(121)
    plt.title(r'cumulative distribution function for $Y = 0$')

    cdf_0 = np.array([np.count_nonzero(x_0 < i) for i in domain], dtype=float)
    cdf_0 /= cdf_0.max()

    plt.plot(domain, cdf_0, label=r'$F_{X,0}(x)$ measured')
    plt.plot(domain, 2 * domain - np.square(domain), '--', label=r'$F_{X,0}(x) = 2x - x^2$')
    plt.legend()

    plt.subplot(122)
    plt.title(r'cumulative distribution function for $Y = 1$')

    cdf_1 = np.array([np.count_nonzero(x_1 < i) for i in domain], dtype=float)
    cdf_1 /= cdf_1.max()

    plt.plot(domain, cdf_1, label=r'$F_{X,1}(x)$ measured')
    plt.plot(domain, np.square(domain), '--', label=r'$F_{X,1}(x) = x^2$')
    plt.legend()

    fig.tight_layout()
    plt.show()


def plot_error_rate_stats(types, dfs, thresholds):
    """
    Plot mean and std of the error rate for different batch sizes
    :param types: string -- string -- Classifier type
    :param dfs: panda.df -- Data frame with mean and std of the error rate for different batch sizes
    :param thresholds: list -- List with thresholds being evaluated
    :return:
    """
    continuous_ts = np.linspace(0, 1, 100)
    for type, df in zip(types, dfs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(f"Rule {type}")

        # First plot: Error rate mean confirms true error
        plt.sca(ax1)
        plt.ylim(0, 100)
        plt.title("Error rate mean [%]")
        df["Mean"].plot(marker="o", yerr=df["Std"], lw=1, ax=ax1)
        plt.plot(continuous_ts, error_threshold_classifier(type=type, analytical=True, threshold=continuous_ts), "k--")
        plt.xticks([0] + thresholds + [1])

        # Second plot: Error rate std declines with more samples
        plt.sca(ax2)
        plt.title("Error rate std [%]")
        df["Std"].T.plot(marker="o", ax=ax2, logx=True, logy=True)

        plt.tight_layout()
