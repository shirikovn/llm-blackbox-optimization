import numpy as np


class Rastrigin:
    def f(self, x):

        A = 10

        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def grad(self, x):

        A = 10

        return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
