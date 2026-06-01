"""
benchmarks.py — стандартные тестовые функции для непрерывной оптимизации.

Каждая функция представлена объектом Benchmark с полями:
    name:        имя
    f(x):        значение функции, x shape (n,) или (B, n)
    grad(x):     градиент, та же форма
    x_star:      аналитический минимум для размерности n
    f_star:      значение в минимуме
    domain:      разумная область старта (lo, hi)

Также есть обёртки:
    ShiftedBenchmark    — сдвинутый минимум: tilde_f(x) = f(x - c)
    RotatedBenchmark    — ортогональное преобразование: tilde_f(x) = f(Q(x - c))

Эти обёртки понадобятся в Блоке 1 для shifted-experiment.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Benchmark:
    name: str
    f: Callable[[np.ndarray], np.ndarray]
    grad: Callable[[np.ndarray], np.ndarray]
    x_star_fn: Callable[[int], np.ndarray]  # принимает n, возвращает минимум
    f_star: float
    domain: tuple[float, float]

    def x_star(self, n: int) -> np.ndarray:
        return self.x_star_fn(n)


# -----------------------------------------------------------------------------
# Sphere: f(x) = sum x_i^2,  min at 0
# -----------------------------------------------------------------------------
def _sphere_f(x):
    return np.sum(x ** 2, axis=-1)


def _sphere_grad(x):
    return 2.0 * x


SPHERE = Benchmark(
    name="sphere",
    f=_sphere_f,
    grad=_sphere_grad,
    x_star_fn=lambda n: np.zeros(n),
    f_star=0.0,
    domain=(-5.0, 5.0),
)


# -----------------------------------------------------------------------------
# Rosenbrock: классический 2D — min at (1,1); n-мерное обобщение — min at (1,...,1)
# f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
# -----------------------------------------------------------------------------
def _rosenbrock_f(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1.0 - x[..., :-1]) ** 2, axis=-1)


def _rosenbrock_grad(x):
    x = np.atleast_1d(x).astype(float)
    g = np.zeros_like(x)
    g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


ROSENBROCK = Benchmark(
    name="rosenbrock",
    f=_rosenbrock_f,
    grad=_rosenbrock_grad,
    x_star_fn=lambda n: np.ones(n),
    f_star=0.0,
    domain=(-2.0, 2.0),
)


# -----------------------------------------------------------------------------
# Rastrigin: f(x) = 10n + sum [x_i^2 - 10 cos(2 pi x_i)], min at 0
# Многомодальная, много локальных минимумов на регулярной решётке.
# -----------------------------------------------------------------------------
def _rastrigin_f(x):
    n = x.shape[-1]
    return 10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def _rastrigin_grad(x):
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


RASTRIGIN = Benchmark(
    name="rastrigin",
    f=_rastrigin_f,
    grad=_rastrigin_grad,
    x_star_fn=lambda n: np.zeros(n),
    f_star=0.0,
    domain=(-5.12, 5.12),
)


# -----------------------------------------------------------------------------
# Ackley: f(x) = -20 exp(-0.2 sqrt(mean(x^2))) - exp(mean(cos(2 pi x))) + 20 + e
# min at 0; глубокая воронка в окружении плоского ландшафта.
# -----------------------------------------------------------------------------
def _ackley_f(x):
    n = x.shape[-1]
    mean_sq = np.mean(x ** 2, axis=-1)
    mean_cos = np.mean(np.cos(2.0 * np.pi * x), axis=-1)
    return -20.0 * np.exp(-0.2 * np.sqrt(mean_sq)) - np.exp(mean_cos) + 20.0 + np.e


def _ackley_grad(x):
    n = x.shape[-1]
    eps = 1e-12
    mean_sq = np.mean(x ** 2)
    s = np.sqrt(mean_sq + eps)
    mean_cos = np.mean(np.cos(2.0 * np.pi * x))
    # d/dx_i of -20 exp(-0.2 s):
    d_first = -20.0 * np.exp(-0.2 * s) * (-0.2) * (x / (n * s))
    # d/dx_i of -exp(mean_cos):
    d_second = -np.exp(mean_cos) * (-2.0 * np.pi / n) * np.sin(2.0 * np.pi * x)
    return d_first + d_second


ACKLEY = Benchmark(
    name="ackley",
    f=_ackley_f,
    grad=_ackley_grad,
    x_star_fn=lambda n: np.zeros(n),
    f_star=0.0,
    domain=(-5.0, 5.0),
)


# -----------------------------------------------------------------------------
# Griewank: f(x) = 1 + sum(x_i^2)/4000 - prod cos(x_i / sqrt(i))
# min at 0; квази-выпуклая на больших масштабах с мелкими осцилляциями.
# -----------------------------------------------------------------------------
def _griewank_f(x):
    n = x.shape[-1]
    idx = np.arange(1, n + 1)
    sum_part = np.sum(x ** 2, axis=-1) / 4000.0
    prod_part = np.prod(np.cos(x / np.sqrt(idx)), axis=-1)
    return 1.0 + sum_part - prod_part


def _griewank_grad(x):
    n = x.shape[-1]
    idx = np.arange(1, n + 1)
    # d sum_part / dx_i = 2 x_i / 4000
    g_sum = 2.0 * x / 4000.0
    # d prod_part / dx_i = - (1/sqrt(i)) sin(x_i / sqrt(i)) * prod_{j != i} cos(x_j / sqrt(j))
    cos_vec = np.cos(x / np.sqrt(idx))
    sin_vec = np.sin(x / np.sqrt(idx))
    full_prod = np.prod(cos_vec)
    # делим аккуратно — может быть ноль
    g_prod = np.zeros_like(x)
    for i in range(n):
        others = full_prod / cos_vec[i] if abs(cos_vec[i]) > 1e-12 else np.prod(
            np.delete(cos_vec, i)
        )
        g_prod[i] = -(1.0 / np.sqrt(idx[i])) * sin_vec[i] * others
    # f = 1 + sum - prod → df/dx = g_sum - g_prod
    return g_sum - g_prod


GRIEWANK = Benchmark(
    name="griewank",
    f=_griewank_f,
    grad=_griewank_grad,
    x_star_fn=lambda n: np.zeros(n),
    f_star=0.0,
    domain=(-10.0, 10.0),
)


ALL_BENCHMARKS = {
    "sphere": SPHERE,
    "rosenbrock": ROSENBROCK,
    "rastrigin": RASTRIGIN,
    "ackley": ACKLEY,
    "griewank": GRIEWANK,
}


# =============================================================================
# Обёртки для Блока 1 (shifted/rotated)
# =============================================================================
class ShiftedBenchmark:
    """tilde_f(x) = f(x - c).  Сохраняет все свойства (выпуклость, овраг и т.д.),
    но минимум смещён в c + x_star(base)."""

    def __init__(self, base: Benchmark, shift: np.ndarray):
        self.base = base
        self.shift = np.asarray(shift, dtype=float)
        self.name = f"{base.name}_shifted"
        self.f_star = base.f_star
        self.domain = base.domain

    def f(self, x):
        return self.base.f(x - self.shift)

    def grad(self, x):
        return self.base.grad(x - self.shift)

    def x_star(self, n: int):
        return self.shift + self.base.x_star_fn(n)


class RotatedBenchmark:
    """tilde_f(x) = f(Q (x - c)).  Q — ортогональная матрица.  Минимум по-прежнему в c."""

    def __init__(self, base: Benchmark, Q: np.ndarray, shift: Optional[np.ndarray] = None):
        self.base = base
        self.Q = np.asarray(Q, dtype=float)
        self.shift = np.zeros(self.Q.shape[0]) if shift is None else np.asarray(shift, dtype=float)
        self.name = f"{base.name}_rotated"
        self.f_star = base.f_star
        self.domain = base.domain

    def f(self, x):
        return self.base.f((x - self.shift) @ self.Q.T)

    def grad(self, x):
        return self.Q.T @ self.base.grad((x - self.shift) @ self.Q.T)

    def x_star(self, n: int):
        return self.shift + self.base.x_star_fn(n)


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Случайная ортогональная матрица через QR-разложение."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # фиксируем знак, чтобы Q был детерминирован при том же rng
    Q = Q * np.sign(np.diag(R))
    return Q
