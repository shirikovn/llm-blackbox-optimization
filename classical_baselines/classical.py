"""
classical.py — классические методы оптимизации.

Каждый метод имеет одинаковый интерфейс:
    optimizer(f, grad, x0, K, **hparams) → (xs, fs)
где
    xs : np.ndarray shape (K+1, n) — траектория, включая x0
    fs : np.ndarray shape (K+1,)   — значения f вдоль траектории

Это позволяет логировать всё единообразно (см. metrics.py).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


def gradient_descent(f, grad, x0, K, lr=0.01):
    x = np.asarray(x0, dtype=float).copy()
    xs = [x.copy()]
    fs = [float(f(x))]
    for k in range(K):
        x = x - lr * grad(x)
        xs.append(x.copy())
        fs.append(float(f(x)))
    return np.array(xs), np.array(fs)


def heavy_ball(f, grad, x0, K, lr=0.01, beta=0.9):
    """Polyak momentum: x_{k+1} = x_k - lr * grad(x_k) + beta * (x_k - x_{k-1})."""
    x = np.asarray(x0, dtype=float).copy()
    x_prev = x.copy()
    xs = [x.copy()]
    fs = [float(f(x))]
    for k in range(K):
        g = grad(x)
        x_new = x - lr * g + beta * (x - x_prev)
        x_prev = x
        x = x_new
        xs.append(x.copy())
        fs.append(float(f(x)))
    return np.array(xs), np.array(fs)


def nesterov(f, grad, x0, K, lr=0.01, beta=0.9):
    """Nesterov accelerated gradient (look-ahead form)."""
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    xs = [x.copy()]
    fs = [float(f(x))]
    for k in range(K):
        # look-ahead point
        y = x + beta * v
        g = grad(y)
        v_new = beta * v - lr * g
        x = x + v_new
        v = v_new
        xs.append(x.copy())
        fs.append(float(f(x)))
    return np.array(xs), np.array(fs)


def adam(f, grad, x0, K, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    xs = [x.copy()]
    fs = [float(f(x))]
    for k in range(1, K + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        xs.append(x.copy())
        fs.append(float(f(x)))
    return np.array(xs), np.array(fs)


def adamw(f, grad, x0, K, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """AdamW = Adam + weight decay (Loshchilov & Hutter, 2019).

    На наших тестовых функциях weight_decay тянет траекторию к нулю — что для
    Sphere/Rastrigin/Ackley/Griewank (минимум в 0) выгодно, а для Rosenbrock
    (минимум в (1,...,1)) и для сдвинутых функций — наоборот. Это часть
    сравнения: AdamW по-разному ведёт себя в зависимости от того, где минимум."""
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    xs = [x.copy()]
    fs = [float(f(x))]
    for k in range(1, K + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * x
        xs.append(x.copy())
        fs.append(float(f(x)))
    return np.array(xs), np.array(fs)


def bfgs(f, grad, x0, K, lr=None):
    """BFGS через scipy.  lr игнорируется (line search внутри).

    Возвращаем K+1 точек траектории: scipy callback даёт промежуточные x.
    Если фактических итераций меньше K — повторяем последнюю точку, чтобы
    траектория имела одинаковую длину со всеми остальными (так удобнее
    строить общие таблицы)."""
    x = np.asarray(x0, dtype=float).copy()
    xs = [x.copy()]
    fs = [float(f(x))]

    def callback(xk):
        xs.append(np.asarray(xk).copy())
        fs.append(float(f(xk)))

    minimize(
        f,
        x0,
        jac=grad,
        method="BFGS",
        callback=callback,
        options={"maxiter": K, "gtol": 0.0, "disp": False},
    )

    # дополняем до K+1, если scipy сошёлся раньше
    while len(xs) < K + 1:
        xs.append(xs[-1].copy())
        fs.append(fs[-1])
    # обрезаем если каким-то образом стало больше
    xs = xs[: K + 1]
    fs = fs[: K + 1]
    return np.array(xs), np.array(fs)


ALL_OPTIMIZERS = {
    "gd": gradient_descent,
    "heavy_ball": heavy_ball,
    "nesterov": nesterov,
    "adam": adam,
    "adamw": adamw,
    "bfgs": bfgs,
}


# -----------------------------------------------------------------------------
# Подбор learning rate: простой grid search
# -----------------------------------------------------------------------------
def grid_search_lr(
    optimizer_name: str,
    f,
    grad,
    x0,
    K,
    lrs: list[float] | None = None,
    extra_hparams: dict | None = None,
) -> tuple[float, float]:
    """Возвращает (best_lr, best_final_f) — lr, дающий минимальный финальный f.

    Тонкость: некоторые комбинации (Adam на Rosenbrock с большим lr) дают NaN
    или overflow. Такие запуски отбрасываем, predupreждения подавляем."""
    if lrs is None:
        # 21 точка: 1e-5 ... 1.0 — достаточно широкая сетка
        lrs = np.logspace(-5, 0, 21).tolist()
    extra = extra_hparams or {}
    opt = ALL_OPTIMIZERS[optimizer_name]

    best_lr, best_final = lrs[0], np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        for lr in lrs:
            try:
                _, fs = opt(f, grad, x0, K, lr=lr, **extra)
                final = fs[-1]
                if np.isfinite(final) and final < best_final:
                    best_final, best_lr = float(final), lr
            except (FloatingPointError, OverflowError, ValueError):
                continue
    return best_lr, best_final
