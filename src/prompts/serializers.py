import numpy as np


def format_float(x):

    return f"{x:+.3e}"


def format_vector(v):

    return "[" + ", ".join(format_float(z) for z in v) + "]"


def format_grad(grad, grad_format):

    if grad_format == "raw":
        return [f"grad(x) = {format_vector(grad)}"]

    if grad_format == "decomposed":
        g = np.asarray(grad)

        norm = float(np.linalg.norm(g))

        if norm > 0:
            direction = g / norm

        else:
            direction = g

        return [
            f"||grad(x)|| = {norm:+.3e}",
            f"grad_direction(x) = {format_vector(direction)}",
        ]

    raise ValueError(f"unknown grad_format: {grad_format}")


def format_return_hint(n):

    return "[" + ", ".join(["float"] * n) + "]"


def build_prompt(
    history,
    use_gradient=True,
    history_size=5,
    grad_format="raw",
):

    history = history[-history_size:]

    n = len(history[0]["x"])

    lines = []

    lines.append("Minimize the function.")

    lines.append("")

    for i, item in enumerate(history):
        lines.append(f"Step {i}")

        lines.append(f"x = {format_vector(item['x'])}")

        lines.append(f"f(x) = {item['f']:.6e}")

        if use_gradient:
            lines.extend(format_grad(item["grad"], grad_format))

        lines.append("")

    lines.append("Propose the next point.")

    lines.append("Return ONLY:")

    lines.append(format_return_hint(n))

    return "\n".join(lines)
