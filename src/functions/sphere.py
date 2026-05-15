import numpy as np


class Sphere:
    def f(self, x):
        return np.sum(x**2)

    def grad(self, x):
        return 2 * x
