import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_2d(
    function,
    trajectory,
    save_path,
    resolution=400,
    padding_ratio=0.15,
    outlier_percentile=90,
    max_outlier_distance=5.0,
    levels=60,
):

    # =====================================================
    # TRAJECTORY
    # =====================================================

    traj = np.array([
        step["x"]
        for step in trajectory
    ])

    xs_traj = traj[:, 0]
    ys_traj = traj[:, 1]

    # =====================================================
    # ROBUST RANGE
    #
    # Ignore huge hallucinated outliers when
    # computing viewport.
    # =====================================================

    center_x = np.median(xs_traj)
    center_y = np.median(ys_traj)

    distances = np.sqrt(
        (xs_traj - center_x) ** 2
        + (ys_traj - center_y) ** 2
    )

    robust_radius = np.percentile(
        distances,
        outlier_percentile,
    )

    robust_radius = max(
        robust_radius,
        1e-3,
    )

    robust_radius *= max_outlier_distance

    mask = distances <= robust_radius

    robust_x = xs_traj[mask]
    robust_y = ys_traj[mask]

    # fallback
    if len(robust_x) < 2:

        robust_x = xs_traj
        robust_y = ys_traj

    # =====================================================
    # DYNAMIC LIMITS
    # =====================================================

    xmin = robust_x.min()
    xmax = robust_x.max()

    ymin = robust_y.min()
    ymax = robust_y.max()

    xrange = xmax - xmin
    yrange = ymax - ymin

    xrange = max(xrange, 1e-2)
    yrange = max(yrange, 1e-2)

    xmin -= padding_ratio * xrange
    xmax += padding_ratio * xrange

    ymin -= padding_ratio * yrange
    ymax += padding_ratio * yrange

    # make square viewport
    size = max(
        xmax - xmin,
        ymax - ymin,
    )

    xcenter = 0.5 * (xmin + xmax)
    ycenter = 0.5 * (ymin + ymax)

    xmin = xcenter - size / 2
    xmax = xcenter + size / 2

    ymin = ycenter - size / 2
    ymax = ycenter + size / 2

    # =====================================================
    # GRID
    # =====================================================

    xs = np.linspace(
        xmin,
        xmax,
        resolution,
    )

    ys = np.linspace(
        ymin,
        ymax,
        resolution,
    )

    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)

    # =====================================================
    # FUNCTION EVAL
    # =====================================================

    for i in range(resolution):

        for j in range(resolution):

            point = np.array([
                X[i, j],
                Y[i, j],
            ])

            value = function.f(point)

            # numerical stability
            if np.isnan(value):
                value = 1e6

            if np.isinf(value):
                value = 1e6

            Z[i, j] = value

    # =====================================================
    # BETTER CONTOURS
    #
    # Use logarithmic contour spacing
    # for much denser contours near optimum.
    # =====================================================

    zmin = np.min(Z)
    zmax = np.max(Z)

    zmin = max(zmin, 1e-12)

    flat_z = Z.flatten()

    percentiles = np.linspace(
        0,
        100,
        levels,
    )

    contour_levels = np.percentile(
        flat_z,
        percentiles,
    )

    # remove duplicates
    contour_levels = np.unique(
        contour_levels
    )

    # =====================================================
    # PLOT
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(9, 8),
    )

    contour = ax.contour(
        X,
        Y,
        Z,
        levels=contour_levels,
        linewidths=0.8,
    )

    ax.clabel(
        contour,
        inline=True,
        fontsize=7,
    )

    # =====================================================
    # DRAW TRAJECTORY
    # =====================================================

    ax.plot(
        xs_traj,
        ys_traj,
        marker="o",
        linewidth=2,
        markersize=5,
    )

    # start point
    ax.scatter(
        xs_traj[0],
        ys_traj[0],
        s=120,
        marker="s",
        label="start",
    )

    # final point
    ax.scatter(
        xs_traj[-1],
        ys_traj[-1],
        s=120,
        marker="*",
        label="final",
    )

    # =====================================================
    # MARK OUTLIERS
    # =====================================================

    outlier_mask = ~mask

    if np.any(outlier_mask):

        ax.scatter(
            xs_traj[outlier_mask],
            ys_traj[outlier_mask],
            marker="x",
            s=80,
            linewidths=2,
            label="outliers",
        )

    # =====================================================
    # STYLE
    # =====================================================

    ax.set_xlim(
        xmin,
        xmax,
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_aspect("equal")

    ax.set_title(
        "Optimization trajectory"
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.legend()

    ax.grid(
        alpha=0.2,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
