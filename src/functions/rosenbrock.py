import numpy as np


class Rosenbrock:
    def __init__(self, shift=None):

        self.shift = shift

    def _shifted(self, x):

        if self.shift is None:
            return x

        return x - np.asarray(self.shift, dtype=float)

    def f(self, x):

        y = self._shifted(x)

        return 100 * (y[1] - y[0] ** 2) ** 2 + (1 - y[0]) ** 2

    def grad(self, x):

        y = self._shifted(x)

        dx = -400 * y[0] * (y[1] - y[0] ** 2) - 2 * (1 - y[0])
        dy = 200 * (y[1] - y[0] ** 2)

        return np.array([dx, dy])