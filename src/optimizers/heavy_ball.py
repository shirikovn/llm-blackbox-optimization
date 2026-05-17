import numpy as np


class HeavyBall:
    def __init__(
        self,
        logger,
        lr=0.001,
        beta=0.9,
    ):

        self.logger = logger

        self.lr = lr
        self.beta = beta

    def step(
        self,
        history,
        step_id,
    ):

        last = history[-1]

        x = last["x"]
        grad = last["grad"]

        if len(history) >= 2:
            momentum = x - history[-2]["x"]

        else:
            momentum = np.zeros_like(x)

        return x - self.lr * grad + self.beta * momentum
