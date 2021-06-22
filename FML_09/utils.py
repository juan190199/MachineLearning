

def to_img(img, X_new, y_new):
    """

    :param img:
    :param y_new:
    :return:
    """
    img_regressed = img[:, :]
    img_regressed[X_new[:, 0], X_new[:, 1]] = y_new

    return img_regressed
