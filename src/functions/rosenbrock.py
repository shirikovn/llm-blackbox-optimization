import numpy as np


class Rosenbrock:
    def f(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def grad(self, x):
        dx = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        dy = 200 * (x[1] - x[0] ** 2)

        return np.array([dx, dy])
