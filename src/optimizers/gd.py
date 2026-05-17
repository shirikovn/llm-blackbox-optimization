class GradientDescent:
    def __init__(
        self,
        logger,
        lr=0.001,
    ):

        self.logger = logger

        self.lr = lr

    def step(
        self,
        history,
        step_id,
    ):

        last = history[-1]

        return last["x"] - self.lr * last["grad"]
