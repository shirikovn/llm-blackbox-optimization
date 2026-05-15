import json

from pathlib import Path
from datetime import datetime

from omegaconf import OmegaConf

from src.loggers.base import BaseLogger


class FilesystemLogger(BaseLogger):
    def __init__(
        self,
        root_dir="outputs",
    ):

        now = datetime.now()

        self.run_dir = (
            Path(root_dir) / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.prompt_dir = self.run_dir / "prompts"

        self.response_dir = self.run_dir / "responses"

        self.error_dir = self.run_dir / "errors"

        self.plot_dir = self.run_dir / "plots"

        self.plot_dir.mkdir(
            exist_ok=True,
        )

        for d in [
            self.prompt_dir,
            self.response_dir,
            self.error_dir,
        ]:
            d.mkdir(
                exist_ok=True,
            )

        self.trajectory = []
        self.metrics = {}

    def log_prompt(
        self,
        step,
        prompt,
    ):

        path = self.prompt_dir / f"step_{step:03d}.txt"

        with open(path, "w") as f:
            f.write(prompt)

    def get_plot_path(
        self,
        filename,
    ):

        return self.plot_dir / filename

    def log_response(
        self,
        step,
        response,
    ):

        path = self.response_dir / f"step_{step:03d}.txt"

        with open(path, "w") as f:
            f.write(response)

    def log_step(
        self,
        step,
        x,
        fx,
        grad,
    ):

        self.trajectory.append(
            {
                "step": step,
                "x": x.tolist(),
                "f": float(fx),
                "grad": grad.tolist(),
            }
        )

    def log_metric(
        self,
        name,
        value,
    ):

        self.metrics[name] = value

    def log_error(
        self,
        error,
    ):

        path = self.error_dir / "errors.txt"

        with open(path, "a") as f:
            f.write(str(error))
            f.write("\n")

    def save_config(
        self,
        config,
    ):

        path = self.run_dir / "config.yaml"

        OmegaConf.save(
            config=config,
            f=path,
        )

    def finalize(self):

        trajectory_path = self.run_dir / "trajectory.json"

        with open(
            trajectory_path,
            "w",
        ) as f:
            json.dump(
                self.trajectory,
                f,
                indent=2,
            )

        metrics_path = self.run_dir / "metrics.json"

        with open(
            metrics_path,
            "w",
        ) as f:
            json.dump(
                self.metrics,
                f,
                indent=2,
            )
