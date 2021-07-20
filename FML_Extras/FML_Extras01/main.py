import numpy as np
import sympy as sp

from gaussian_newton import GaussianNewton


def main():
    ds = GaussianNewton(
        expr="(b1 + (b2 * x) + (b3 * (x ** 2)) + (b4 * (x ** 3))) / ( 1 + (b5 * x) + (b6 * (x ** 2)) + (b7 * (x ** 3)))",
        symbols=sp.symbols("x b1:8"),
        X=np.array((
            -3.067, -2.981, -2.921, -2.912, -2.840, -2.797, -2.702, -2.699,
            -2.633, -2.481, -2.363, -2.322, -1.501, -1.460, -1.274, -1.212,
            -1.100, -1.046, -0.915, -0.714, -0.566, -0.545, -0.400, -0.309,
            -0.109, -0.103, 0.010, 0.119, 0.377, 0.790, 0.963, 1.006,
            1.115, 1.572, 1.841, 2.047, 2.200
        )),
        y=np.array((
            80.574, 84.248, 87.264, 87.195, 89.076, 89.608,
            89.868, 90.101, 92.405, 95.854, 100.696, 101.060,
            401.672, 390.724, 567.534, 635.316, 733.054, 759.087,
            894.206, 990.785, 1090.109, 1080.914, 1122.643, 1178.351,
            1260.531, 1273.514, 1288.339, 1327.543, 1353.863, 1414.509,
            1425.208, 1421.384, 1442.962, 1464.350, 1468.705, 1447.894,
            1457.628
        )),
        cvals=np.array((
            1.2881396800e+03, 1.4910792535e+03, 5.8323836877e+02,
            7.5416644291e+01, 9.6629502864e-01, 3.9797285797e-01,
            4.9727297349e-02
        )),
        init_guess=np.array(((1000, 1000, 400, 40, 0.7, 0.3, 0.03),
                             (1300, 1500, 500, 75, 1.0, 0.4, 0.05)))
    )

    # Inhibit wrapping of arrays in print
    np.set_printoptions(ds.init_guess_)

    for i, init_guess in enumerate(ds.init_guess_):
        sol, its = ds.fit(init_guess)
        cv = ds.cvals
        print("{}, start {}:".format(ds, i + 1))
        print("  Iterations : {}".format(its))
        print("  Calculated : {}".format(sol))
        print("  Certified  : {}".format(cv))
        print("  Difference : {}".format(np.abs(sol - cv)))


if __name__ == '__main__':
    main()
