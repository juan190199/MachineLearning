import numpy as np

from sklearn import model_selection

from data.my_dataset import MyDataset


def mean_squared_error(prediction, target):
    """

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape
    N = prediction.shape[0]
    return np.sum(np.square(prediction - target)) / N


def k_fold_cv(model, feats, targets, k=10):
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


def test_data(models, oerrors, data, n_shuffle_sets, pred_variable='rating'):
    """

    :param data:
    :param n_shuffle_sets:
    :param pred_variable:
    :return:
    """
    shuffle_sets = []
    # Random seed to make results reproducible
    np.random.seed(12345)
    # Shuffling works in place, therefore copy is retrieved
    rating = np.array(data[pred_variable]).copy()
    for i in range(n_shuffle_sets):
        np.random.shuffle(rating)
        shuffled_data = data.copy()
        shuffled_data[pred_variable] = rating
        shuffle_sets.append(MyDataset(shuffled_data))

    # Evaluate shuffle sets
    oerrors = np.array([oerrors]).T
    shuffled_errors = np.empty((len(models), n_shuffle_sets))
    for i, data_set in enumerate(shuffle_sets):
        print("Run shuffle set no. %i" % (i + 1))
        feats = data_set.get_X_oc_()
        targets = data_set.targets
        assert feats.shape[0] == targets.shape[0]
        for j, model in enumerate(models):
            err_train, std_err_train, err_test, std_err_test = k_fold_cv(model, feats, targets)
            shuffled_errors[j, i] = err_test

    n_greater_errors_models = np.sum((shuffled_errors > oerrors), axis=1)

    return n_greater_errors_models
