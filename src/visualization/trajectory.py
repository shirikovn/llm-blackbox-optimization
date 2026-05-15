import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_2d(
    function,
    trajectory,
    save_path,
    xlim=(-5, 5),
    ylim=(-5, 5),
    resolution=200,
):

    xs = np.linspace(
        xlim[0],
        xlim[1],
        resolution,
    )

    ys = np.linspace(
        ylim[0],
        ylim[1],
        resolution,
    )

    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            point = np.array(
                [
                    X[i, j],
                    Y[i, j],
                ]
            )

            Z[i, j] = function.f(point)

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    contour = ax.contour(
        X,
        Y,
        Z,
        levels=30,
    )

    ax.clabel(
        contour,
        inline=True,
        fontsize=8,
    )

    traj = np.array([step["x"] for step in trajectory])

    ax.plot(
        traj[:, 0],
        traj[:, 1],
        marker="o",
    )

    ax.set_title("Optimization trajectory")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()
