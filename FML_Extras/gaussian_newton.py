import numpy as np
import simpy as sp


class GaussianNewton:
    def __init__(self, expr, symbols, X, y, cvals, init_guess):
        """

        :param expr: string
            Representation of dataset's model in a format understood by simpy.sympify

        :param symbols: tuple or list
            SimPy Symbols found in 'expr'.
            The firs one should be the predictor variable and the rest are interpreted as model parameters

        :param X: ndarray
            Design matrix

        :param y: ndarray
            Response variable

        :param cvals: ndarray
            Certified values for model parameters

        :param init_guess: ndarray
            Nested set of initial guesses or starting estimates for the least squares solution
        """
        self.init_guess_ = init_guess
        self.symbols_ = symbols

        self.X_, self.y_, self.cvals = X, y, cvals

        self._x, self._b = symbols[0], symbols[1:]

        # SimPy expressions
        self._symexpr = sp.sympify(expr)
        # Numpy expression
        self._numexpr = sp.lambdify((self._x,) + self._b, self.symexpr, 'numpy')
        # Partial derivatives
        self._pderivs = [self._symexpr.diff(b) for b in self._b]

    def hypothesis(self, X=None, b=None):
        """

        :param X:
        :param b:
        :return:
        """
        if X is None: X = self.X_
        if b is None: b = self.cvals
        return self._numexpr(X, *b)

    def residual(self, b):
        """

        :param b:
        :return:
        """
        X, y = self.X_, self.y_
        return y - self._numexpr(X, *b)

    def jacobian(self, b):
        # Substitute parameter in partial derivatives
        subs = [pd.subs(zip(self._b, b)) for pd in self._pderivs]
        # Evaluate substituted partial derivatives for all x values
        vals = [sp.lambdify(self._x, sub, 'numpy')(self.X_) for sub in subs]
        # Arrange values in column-major order
        return np.column_stack(vals)

    def fit(self, init_guess, tol=1e-10, maxits=256):
        """

        :param init_guess: tuple, list or ndarray
            Initial guess for the algorithm

        :param tol: float, default=1e-10
            Tolerance threshold

        :param maxits: int, default=256
            Maximum number of iterations

        :return:
        """
        dx = np.ones(len(init_guess))
        xn = np.array(init_guess)

        i = 0
        while (i < maxits) and (dx[dx > tol].size > 0):
            dx = np.dot(np.linalg.pinv(self.jacobian(xn)), - self.residual(xn))
            xn += dx
            i += 1

        return xn, i
