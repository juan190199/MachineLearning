import numpy as np

from sklearn import model_selection


def mean_squared_error(prediction, target):
    """

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape
    N = prediction.shape[0]
    return np.sum(np.square(prediction - target)) / N


def k_fold_cv(model, feats, targets, k):
    """

    :param model:
    :param feats:
    :param targets:
    :param k:
    :return:
    """
    assert targets.shape[0] == feats.shape[0]
    kf = model_selection.KFold(n_splits=k)
    err_train, err_test = [], []
    for train, test in kf.split(feats):
        model.fit(feats[train], targets[train])
        err_train.append(mean_squared_error(model.predict(feats[train]), targets[train]))
        err_test.append(mean_squared_error(model.predict(feats[test]), targets[test]))
    return np.mean(err_train), np.std(err_train), np.mean(err_test), np.std(err_test)
