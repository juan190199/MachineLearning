import matplotlib.pyplot as plt


def plot_gn(x, y, yn, fit, residual):
    """

    :param x:
    :param y:
    :param yn:
    :param fit:
    :param residual:
    :return:
    """
    plt.figure()
    plt.plot(x, y, label="Original, noiseless signal", linewidth=2)
    plt.plot(x, yn, label="Noisy signal", linewidth=2)
    plt.plot(x, fit, label="Fit", linewidth=2)
    plt.plot(x, residual, label="Residual", linewidth=2)
    plt.title("Gauss-Newton: curve fitting example")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()
