import numpy as np


class Ackley:
    def __init__(self, a=20.0, b=0.2, c=2 * np.pi):

        self.a = a
        self.b = b
        self.c = c

    def f(self, x):

        n = len(x)

        mean_sq = np.sum(x**2) / n
        mean_cos = np.sum(np.cos(self.c * x)) / n

        return (
            -self.a * np.exp(-self.b * np.sqrt(mean_sq))
            - np.exp(mean_cos)
            + self.a
            + np.e
        )

    def grad(self, x):

        n = len(x)

        mean_sq = np.sum(x**2) / n
        r = np.sqrt(mean_sq)

        mean_cos = np.sum(np.cos(self.c * x)) / n

        if r == 0.0:
            term1 = np.zeros_like(x)

        else:
            term1 = self.a * self.b * np.exp(-self.b * r) * x / (n * r)

        term2 = (self.c / n) * np.exp(mean_cos) * np.sin(self.c * x)

        return term1 + term2
