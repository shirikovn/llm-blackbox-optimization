"""
metrics.py — метрики оценки траектории оптимизатора.

Принимают:
    xs : np.ndarray (K+1, n) — траектория
    fs : np.ndarray (K+1,)   — значения f
    grad_fn :                — для cosine с антиградиентом (опц.)
    f_star : float           — известное значение в минимуме
    x_star : np.ndarray (n,) — известный аргумент минимума (опц., для distance)
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def final_gap(fs: np.ndarray, f_star: float) -> float:
    return float(fs[-1] - f_star)


def best_gap(fs: np.ndarray, f_star: float) -> float:
    """Лучшее значение, достигнутое за всю траекторию (а не только финальное).
    Полезно для нестабильных оптимизаторов (DeepSeek на Rosenbrock)."""
    finite = fs[np.isfinite(fs)]
    if len(finite) == 0:
        return float("inf")
    return float(np.min(finite) - f_star)


def time_to_eps(fs: np.ndarray, f_star: float, eps: float = 1e-3) -> Optional[int]:
    """Шаг, на котором впервые достигнут gap < eps. None — не достигнут."""
    below = np.where(fs - f_star < eps)[0]
    return int(below[0]) if len(below) > 0 else None


def success(fs: np.ndarray, f_star: float, eps: float = 1e-3) -> bool:
    return time_to_eps(fs, f_star, eps) is not None


def trajectory_length(xs: np.ndarray) -> float:
    """sum_k ||x_{k+1} - x_k||."""
    diffs = np.diff(xs, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def cosine_with_antigrad(xs: np.ndarray, grad_fn) -> np.ndarray:
    """Косинус между фактическим шагом dx_k = x_{k+1} - x_k и -grad(x_k).

    Возвращает массив длины K: для каждой пары соседних точек.
    NaN ставится там, где либо градиент нулевой, либо шаг нулевой."""
    K = len(xs) - 1
    out = np.full(K, np.nan)
    for k in range(K):
        dx = xs[k + 1] - xs[k]
        g = grad_fn(xs[k])
        nd, ng = np.linalg.norm(dx), np.linalg.norm(g)
        if nd > 1e-12 and ng > 1e-12:
            out[k] = float(np.dot(dx, -g) / (nd * ng))
    return out


def step_consistency(xs: np.ndarray) -> np.ndarray:
    """Косинус между двумя соседними шагами: cos(dx_k, dx_{k-1}).

    Индикатор устойчивого направленного движения (momentum) vs осцилляций.
    Возвращает массив длины K-1."""
    K = len(xs) - 1
    out = np.full(K - 1, np.nan)
    for k in range(1, K):
        dx1 = xs[k] - xs[k - 1]
        dx2 = xs[k + 1] - xs[k]
        n1, n2 = np.linalg.norm(dx1), np.linalg.norm(dx2)
        if n1 > 1e-12 and n2 > 1e-12:
            out[k - 1] = float(np.dot(dx1, dx2) / (n1 * n2))
    return out


def step_sizes(xs: np.ndarray) -> np.ndarray:
    """||x_{k+1} - x_k|| для каждого k. Длина K."""
    return np.linalg.norm(np.diff(xs, axis=0), axis=1)


def distance_to_minimum(xs: np.ndarray, x_star: np.ndarray) -> np.ndarray:
    """||x_k - x*|| вдоль траектории. Длина K+1."""
    return np.linalg.norm(xs - x_star[None, :], axis=1)


def all_metrics_for_trajectory(
    xs: np.ndarray,
    fs: np.ndarray,
    grad_fn,
    f_star: float,
    x_star: Optional[np.ndarray] = None,
    eps: float = 1e-3,
) -> dict:
    """Собирает все скалярные и векторные метрики в один dict."""
    cos_grad = cosine_with_antigrad(xs, grad_fn)
    cos_step = step_consistency(xs)
    sizes = step_sizes(xs)

    out = {
        "final_gap": final_gap(fs, f_star),
        "best_gap": best_gap(fs, f_star),
        "T_eps": time_to_eps(fs, f_star, eps),
        "success": success(fs, f_star, eps),
        "traj_length": trajectory_length(xs),
        "mean_cosine_antigrad": float(np.nanmean(cos_grad))
        if np.any(np.isfinite(cos_grad))
        else float("nan"),
        "mean_step_consistency": float(np.nanmean(cos_step))
        if np.any(np.isfinite(cos_step))
        else float("nan"),
        "median_step_size": float(np.nanmedian(sizes)),
        "max_step_size": float(np.nanmax(sizes)),
        # Векторные — для рисунков
        "cosine_per_step": cos_grad.tolist(),
        "consistency_per_step": cos_step.tolist(),
        "step_sizes": sizes.tolist(),
    }
    if x_star is not None:
        out["final_dist_to_min"] = float(np.linalg.norm(xs[-1] - x_star))
        out["dist_to_min_per_step"] = distance_to_minimum(xs, x_star).tolist()
    return out
