import numpy as np


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
