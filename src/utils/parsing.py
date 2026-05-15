import ast
import re

import numpy as np


def parse_vector(text):

    match = re.search(
        r"\[([^\]]+)\]",
        text,
    )

    if match is None:
        raise ValueError(f"Cannot parse vector from:\n{text}")

    vec = ast.literal_eval("[" + match.group(1) + "]")

    return np.array(
        vec,
        dtype=float,
    )
