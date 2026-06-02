import numpy as np
import matplotlib.pyplot as plt


def plot_function_preview(
    function,
    save_path,
    resolution=300,
):
    A = getattr(function, "A", np.eye(2))
    b = getattr(function, "b", np.zeros(2))

    base = function.base

    # =====================================================
    # TRUE BASE OPTIMUM
    # =====================================================

    def base_optimum():
        return np.zeros(2)

    z_star = base_optimum()

    # full inverse mapping (IMPORTANT: b INCLUDED)
    x_star = np.linalg.inv(A) @ (z_star - b)

    # =====================================================
    # WINDOW AROUND OPTIMUM ONLY
    # =====================================================

    pad = 3.0

    xs = np.linspace(
        x_star[0] - pad,
        x_star[0] + pad,
        resolution,
    )

    ys = np.linspace(
        x_star[1] - pad,
        x_star[1] + pad,
        resolution,
    )

    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    # =====================================================
    # EVAL FUNCTION
    # =====================================================

    for i in range(resolution):
        for j in range(resolution):

            x = np.array([X[i, j], Y[i, j]])
            val = function.f(x)

            if not np.isfinite(val):
                val = 1e6

            Z[i, j] = val

    # =====================================================
    # PLOT
    # =====================================================

    fig, ax = plt.subplots(figsize=(9, 8))

    levels = np.percentile(Z, np.linspace(0, 95, 40))

    contour = ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        linewidths=0.8,
    )

    ax.clabel(contour, fontsize=6)

    # =====================================================
    # ONLY ONE MARKER (IMPORTANT FIX)
    # =====================================================

    ax.scatter(
        x_star[0],
        x_star[1],
        s=200,
        marker="*",
        color="red",
        label="optimum x*",
        zorder=5,
    )

    ax.text(
        x_star[0],
        x_star[1],
        "  x*",
        color="red",
        fontsize=12,
        weight="bold",
    )

    # =====================================================
    # STYLE CLEANUP
    # =====================================================

    ax.set_title("Transformed Function Preview")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
