"""
llm_interactive.py — LLM-оптимизатор с интерактивным интерфейсом.

Использует промпт V1 (Full/Raw): история из последних `history_len` шагов
с x, f(x), grad(x). При каждом шаге:
  1) скрипт печатает промпт в консоль;
  2) пользователь копирует его в Claude/ChatGPT/Gemini, получает ответ;
  3) вставляет ответ в формате [a, b, ...] обратно в консоль.

Также есть режим `replay`: ответы заранее даны в файле — для воспроизведения
прошлых экспериментов без повторного обращения к модели.
"""

from __future__ import annotations
import json
import re
import sys
import numpy as np
from pathlib import Path


# Точный финальный шаблон V1 из текущего main.tex
PROMPT_HEADER = "Minimize the function."
PROMPT_FOOTER = (
    "Propose the next point.\nReturn ONLY: [float, "
    + "float, " * 10
    + "float] (depending on dim)"
)


def format_vec(x: np.ndarray) -> str:
    """Форматирование в виде, как в финальном промпте: [+2.000e-01, +1.600e+00]"""
    parts = [f"{v:+.3e}" for v in x]
    return "[" + ", ".join(parts) + "]"


def build_prompt(
    history: list[tuple[np.ndarray, float, np.ndarray]], history_len: int = 5
) -> str:
    """history — список (x, f, grad). Берём последние history_len."""
    tail = history[-history_len:]
    lines = [PROMPT_HEADER, ""]
    for i, (x, fv, g) in enumerate(tail):
        step_idx = len(history) - len(tail) + i
        lines.append(f"Step {step_idx}")
        lines.append(f"x = {format_vec(x)}")
        lines.append(f"f(x) = {fv:.6e}")
        lines.append(f"grad(x) = {format_vec(g)}")
        lines.append("")
    lines.append(
        PROMPT_FOOTER.replace(
            "[float, " + "float, " * 10 + "float] (depending on dim)",
            "[" + ", ".join(["float"] * len(history[-1][0])) + "]",
        )
    )
    return "\n".join(lines)


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_response(text: str, n: int) -> np.ndarray:
    """Извлекает n чисел из ответа LLM. Принимает форматы [a, b], a, b, (a, b) и т.п."""
    # сначала пробуем строгий формат [a, b, ...]
    m = re.search(r"\[([^\]]+)\]", text)
    if m:
        nums = _NUM_RE.findall(m.group(1))
        if len(nums) == n:
            return np.array([float(v) for v in nums])
    # fallback: ищем первые n чисел в строке
    nums = _NUM_RE.findall(text)
    if len(nums) < n:
        raise ValueError(f"Could not parse {n} numbers from: {text!r}")
    return np.array([float(v) for v in nums[:n]])


def llm_optimize_interactive(
    f,
    grad,
    x0,
    K,
    *,
    history_len: int = 5,
    log_path: str | Path | None = None,
    max_parse_retries: int = 3,
):
    """Главный цикл интерактивного LLM-оптимизатора.

    Возвращает (xs, fs) тех же форм, что и классические методы, для
    единообразного логирования через metrics.py.

    log_path — куда сохранять историю (json со всеми промптами и ответами),
    чтобы потом можно было воспроизвести через replay-режим."""
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)
    history = [(x.copy(), float(f(x)), grad(x).copy())]
    xs = [x.copy()]
    fs = [history[0][1]]
    log = {"x0": x.tolist(), "n": n, "K": K, "history_len": history_len, "steps": []}

    for k in range(K):
        prompt = build_prompt(history, history_len=history_len)
        print("\n" + "=" * 70)
        print(f"STEP {k + 1}/{K}  —  copy prompt below into your LLM:")
        print("=" * 70)
        print(prompt)
        print("=" * 70)

        x_new = None
        attempts = []
        for attempt in range(max_parse_retries):
            try:
                resp = input(f"\nLLM response (or 'quit' to stop): ").strip()
            except EOFError:
                print("\n[interrupted, saving and exiting]")
                resp = "quit"
            if resp.lower() in ("quit", "q", "exit"):
                # сохраняем то, что есть, и выходим раньше
                if log_path is not None:
                    _save_log(log_path, log)
                return np.array(xs), np.array(fs)
            attempts.append(resp)
            try:
                x_new = parse_response(resp, n)
                break
            except ValueError as e:
                print(f"  [parse error: {e}; retry {attempt + 1}/{max_parse_retries}]")

        if x_new is None:
            print(
                f"  [step {k + 1} failed to parse after {max_parse_retries} attempts]"
            )
            # повторяем последнюю точку как заглушку
            x_new = xs[-1].copy()

        fv = float(f(x_new))
        gv = grad(x_new).copy()
        history.append((x_new.copy(), fv, gv))
        xs.append(x_new.copy())
        fs.append(fv)
        log["steps"].append(
            {
                "k": k + 1,
                "prompt": prompt,
                "attempts": attempts,
                "x": x_new.tolist(),
                "f": fv,
                "grad": gv.tolist(),
            }
        )

        print(f"  → x_{k + 1} = {format_vec(x_new)},  f = {fv:.4e}")

    if log_path is not None:
        _save_log(log_path, log)
    return np.array(xs), np.array(fs)


def llm_optimize_replay(
    f,
    grad,
    x0,
    K,
    *,
    replay_log: str | Path,
):
    """Воспроизводит траекторию из сохранённого лога.  Полезно, если
    нужно перебрать метрики, не общаясь снова с LLM."""
    with open(replay_log) as fh:
        log = json.load(fh)
    xs = [np.asarray(log["x0"], dtype=float)]
    fs = [float(f(xs[0]))]
    for step in log["steps"][:K]:
        x = np.asarray(step["x"], dtype=float)
        xs.append(x)
        fs.append(float(f(x)))
    while len(xs) < K + 1:
        xs.append(xs[-1])
        fs.append(fs[-1])
    return np.array(xs), np.array(fs)


def _save_log(path, log: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print(f"\n[log saved to {p}]")
