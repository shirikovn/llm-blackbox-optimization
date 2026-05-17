import json

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_trajectory(run_dir):

    path = Path(run_dir) / "trajectory.json"

    with open(path) as f:
        return json.load(f)


def overlay_trajectories_2d(
    function,
    run_dirs,
    save_path,
    labels=None,
    xlim=(-5, 5),
    ylim=(-5, 5),
    resolution=200,
):

    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)

    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = function.f(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.contour(X, Y, Z, levels=30, alpha=0.4)

    for k, run_dir in enumerate(run_dirs):
        traj = _load_trajectory(run_dir)

        xs_traj = np.array([s["x"] for s in traj])

        label = labels[k] if labels else Path(run_dir).name

        ax.plot(
            xs_traj[:, 0],
            xs_traj[:, 1],
            marker="o",
            label=label,
        )

    ax.legend()

    ax.set_title("Trajectory comparison")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def overlay_convergence(
    run_dirs,
    save_path,
    labels=None,
):

    fig, ax = plt.subplots(figsize=(8, 5))

    for k, run_dir in enumerate(run_dirs):
        traj = _load_trajectory(run_dir)

        values = [s["f"] for s in traj]

        label = labels[k] if labels else Path(run_dir).name

        ax.plot(values, marker="o", label=label)

    ax.set_yscale("log")

    ax.set_xlabel("Step")
    ax.set_ylabel("f(x)")

    ax.set_title("Convergence comparison")

    ax.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()
