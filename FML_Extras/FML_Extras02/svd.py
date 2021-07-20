try:
    import fbpca

    FBPCA_INSTALLED = True
except ImportError:
    FBPCA_INSTALLED = False

from sklearn import extmath


def compute_svd(X, n_components, n_iter, random_state, engine):
    """

    :param X:
    :param n_components:
    :param n_iter:
    :param random_state:
    :param engine:
    :return:
    """
    # Determine what SVD engine to use
    if engine == 'auto':
        engine = 'sklearn'

    # Compute SVD
    if engine == 'fbpca':
        if FBPCA_INSTALLED:
            U, s, V = fbpca.pca(X, k=n_components, n_iter=n_iter)
        else:
            raise ValueError('fbpca is not installed; please install it if you want to use it')
    elif engine == 'sklearn':
        U, s, V = extmath.randomized_svd(
            X,
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )
    else:
        raise ValueError("Engine has to be one of ('auto', 'fbpca', 'sklearn')")

    U, V = extmath.svd_flip(U, V)

    return U, s, V
