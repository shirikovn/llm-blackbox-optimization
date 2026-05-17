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


def format_header(prompt_template, use_gradient):

    if prompt_template == "base":
        return ["Minimize the function."]

    if prompt_template == "explanatory":
        lines = [
            "You are optimizing a continuous function f: R^n -> R.",
            "You see the history of points you have visited and the value of f at each.",
        ]

        if use_gradient:
            lines.append(
                "You also see the gradient of f at each point."
            )

            lines.append(
                "The gradient points in the direction of steepest ascent -- moving against it decreases f."
            )

        lines.append("Your goal: suggest the next point that minimizes f.")

        return lines

    raise ValueError(f"unknown prompt_template: {prompt_template}")


def build_prompt(
    history,
    use_gradient=True,
    history_size=5,
    grad_format="raw",
    prompt_template="base",
):

    history = history[-history_size:]

    n = len(history[0]["x"])

    lines = []

    lines.extend(format_header(prompt_template, use_gradient))

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
