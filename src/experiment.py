import numpy as np

from src.visualization.trajectory import plot_trajectory_2d
from src.visualization.convergence import plot_convergence


class Experiment:
    def __init__(
        self,
        function,
        optimizer,
        logger,
        config,
    ):

        self.function = function
        self.optimizer = optimizer
        self.logger = logger
        self.config = config

    def run(self):

        x = np.array(
            self.config.x0,
            dtype=float,
        )

        history = []

        for step in range(self.config.experiment.steps):
            fx = self.function.f(x)
            grad = self.function.grad(x)

            history.append(
                {
                    "x": x.copy(),
                    "f": fx,
                    "grad": grad.copy(),
                }
            )

            self.logger.log_step(
                step,
                x,
                fx,
                grad,
            )

            print(f"[{step}] f(x)={fx:.6f}")

            x = self.optimizer.step(
                history,
                step,
            )

        plot_trajectory_2d(
            function=self.function,
            trajectory=history,
            save_path=self.logger.get_plot_path("trajectory.png"),
        )

        plot_convergence(
            trajectory=history,
            save_path=self.logger.get_plot_path("convergence.png"),
        )

        self.logger.finalize()
