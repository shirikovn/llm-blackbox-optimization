import argparse

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.visualization.overlay import (
    overlay_trajectories_2d,
    overlay_convergence,
)


def main():

    parser = argparse.ArgumentParser(
        description="Overlay trajectories and convergence curves from multiple runs.",
    )

    parser.add_argument(
        "runs",
        nargs="+",
        help="Paths to run directories (each containing trajectory.json and config.yaml).",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels, one per run (defaults to run dir basenames).",
    )

    parser.add_argument(
        "--out",
        default="comparison",
        help="Output directory for comparison plots.",
    )

    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=(-5.0, 5.0),
    )

    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(-5.0, 5.0),
    )

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError(
            f"got {len(args.labels)} labels for {len(args.runs)} runs"
        )

    config_path = Path(args.runs[0]) / "config.yaml"

    config = OmegaConf.load(config_path)

    function = instantiate(config.function)

    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_trajectories_2d(
        function=function,
        run_dirs=args.runs,
        save_path=out_dir / "trajectory.png",
        labels=args.labels,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
    )

    overlay_convergence(
        run_dirs=args.runs,
        save_path=out_dir / "convergence.png",
        labels=args.labels,
    )

    print(f"wrote {out_dir / 'trajectory.png'}")
    print(f"wrote {out_dir / 'convergence.png'}")


if __name__ == "__main__":
    main()
