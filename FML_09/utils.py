import numpy as np

from scipy.optimize import least_squares


def to_img(img, X_new, y_new):
    """
    Complete image with predicted pixels y_test for X_test data

    :param img:
    :param y_new:
    :return:
    """
    img_regressed = img[:, :]
    img_regressed[X_new[:, 0], X_new[:, 1]] = y_new

    return img_regressed


def circle_from_points(points):
    """

    :param points:
    :return:
    """
    offset = points[0]
    p = points - offset

    # 2x2 linear equations
    A = 2 * p[1:, :]
    b = p[1:, 0] ** 2 + p[1:, 1] ** 2

    try:
        c = np.dot(np.linalg.inv(A), b)
        r = np.sqrt(c[0] ** 2 + c[1] ** 2)
    except np.linalg.linalg.LinAlgError:
        # If the points are on a straight line, there is no solution
        return np.NaN * np.zeros(2), np.NaN
    else:
        return c + offset, r


def refine_fit(models, method):

    fit_models = []
    for circle in models:

        def cost(circ):
            c, r = circ[:2], circ[2]

            if method == 'algebraic':
                return np.sum(np.linalg.norm(circle['MCS'] - c, axis=1) ** 2 - r ** 2)

            elif method == 'lm':
                return np.sum(np.linalg.norm(circle['MCS'] - c, axis=1) - r)

        # Use the RANSAC result as the best guess
        guess = np.concatenate((circle['c'], [circle['r']]))
        fit = least_squares(cost, guess)
        fit_models.append(fit.x)

    return fit_models
