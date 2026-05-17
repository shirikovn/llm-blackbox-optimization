import numpy as np


class Griewank:
    def f(self, x):

        n = len(x)

        i = np.arange(1, n + 1, dtype=float)

        return (
            1.0
            + np.sum(x**2) / 4000.0
            - np.prod(np.cos(x / np.sqrt(i)))
        )

    def grad(self, x):

        n = len(x)

        i = np.arange(1, n + 1, dtype=float)

        sqrt_i = np.sqrt(i)

        cos_terms = np.cos(x / sqrt_i)
        sin_terms = np.sin(x / sqrt_i)

        grad = np.empty(n)

        for k in range(n):
            mask = np.arange(n) != k

            others = np.prod(cos_terms[mask])

            grad[k] = x[k] / 2000.0 + sin_terms[k] / sqrt_i[k] * others

        return grad
