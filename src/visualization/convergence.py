import matplotlib.pyplot as plt


def plot_convergence(
    trajectory,
    save_path,
):

    values = [step["f"] for step in trajectory]

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    ax.plot(values)

    ax.set_yscale("log")

    ax.set_xlabel("Step")
    ax.set_ylabel("f(x)")

    ax.set_title("Convergence curve")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()
