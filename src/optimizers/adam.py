import numpy as np


class Adam:
    def __init__(
        self,
        logger,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):

        self.logger = logger

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None

    def step(
        self,
        history,
        step_id,
    ):

        last = history[-1]

        x = last["x"]
        grad = last["grad"]

        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        t = step_id + 1

        m_hat = self.m / (1 - self.beta1**t)
        v_hat = self.v / (1 - self.beta2**t)

        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
