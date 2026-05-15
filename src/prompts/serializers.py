def format_float(x):

    return f"{x:+.3e}"


def format_vector(v):

    return "[" + ", ".join(format_float(z) for z in v) + "]"


def build_prompt(
    history,
    use_gradient=True,
    history_size=5,
):

    history = history[-history_size:]

    lines = []

    lines.append("Minimize the function.")

    lines.append("")

    for i, item in enumerate(history):
        lines.append(f"Step {i}")

        lines.append(f"x = {format_vector(item['x'])}")

        lines.append(f"f(x) = {item['f']:.6e}")

        if use_gradient:
            lines.append(f"grad(x) = {format_vector(item['grad'])}")

        lines.append("")

    lines.append("Propose the next point.")

    lines.append("Return ONLY:")

    lines.append("[float, float]")

    return "\n".join(lines)
