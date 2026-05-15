class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, x, grad):

        return x - self.lr * grad
