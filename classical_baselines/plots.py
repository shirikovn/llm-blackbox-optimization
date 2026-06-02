"""
plots.py — все основные визуализации для Блока 2 и далее.

Делает то, что просит фидбек проверяющих:
  - больше визуализаций траекторий;
  - более информативное представление таблиц (heatmap).

Функции:
  plot_convergence_grid    Сетка f(x_k) по функциям × методам.
  plot_trajectories_2d     Сетка траекторий на 2D-контурах (только n=2).
  plot_metrics_heatmap     Heatmap-таблица (function × optimizer) по
                           одной метрике (по умолчанию success rate).
  plot_cosine_per_step     Линейная диаграмма cos с антигр. вдоль шагов.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import ALL_BENCHMARKS


def _group(recs: list[dict]):
    """Группируем по (function, n) → optimizer → список записей."""
    out = {}
    for r in recs:
        key = (r["function"], r["n"])
        out.setdefault(key, {}).setdefault(r["optimizer"], []).append(r)
    return out


# -----------------------------------------------------------------------------
def plot_convergence_grid(recs: list[dict], out_path, dim_filter: int | None = None):
    """Сетка convergence: одна функция = одна колонка, строки — размерности.
    На каждом subplot все оптимизаторы, среднее по стартам с полупрозрачной полосой."""
    groups = _group(recs)
    funcs = sorted(set(k[0] for k in groups.keys()))
    dims = sorted(set(k[1] for k in groups.keys()))
    if dim_filter is not None:
        dims = [d for d in dims if d == dim_filter]

    nrows, ncols = len(dims), len(funcs)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False, sharex=True
    )
    cmap = plt.get_cmap("tab10")
    optimizers_seen = sorted(set(r["optimizer"] for r in recs))
    colors = {opt: cmap(i % 10) for i, opt in enumerate(optimizers_seen)}

    for i, n in enumerate(dims):
        for j, fname in enumerate(funcs):
            ax = axs[i, j]
            opt_dict = groups.get((fname, n), {})
            for opt_name, runs in sorted(opt_dict.items()):
                # Все траектории на одной оси шагов
                K = len(runs[0]["fs"]) - 1
                fs_matrix = np.array([r["fs"] for r in runs])  # (J, K+1)
                f_star = ALL_BENCHMARKS[
                    fname.replace("_shifted", "").replace("_rotated", "")
                ].f_star
                gap = np.clip(fs_matrix - f_star, 1e-16, None)
                mean = np.nanmean(gap, axis=0)
                p10 = np.nanpercentile(gap, 10, axis=0)
                p90 = np.nanpercentile(gap, 90, axis=0)
                steps = np.arange(K + 1)
                c = colors[opt_name]
                ax.plot(steps, mean, label=opt_name, color=c, linewidth=1.4)
                ax.fill_between(steps, p10, p90, color=c, alpha=0.12)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(fname, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"n={n}\ngap to f*", fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel("step", fontsize=9)
            ax.grid(True, alpha=0.3, which="both")
            ax.tick_params(labelsize=8)
    # одна общая легенда
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Convergence: gap to optimum (mean + 10–90% band over J starts)",
        fontsize=11,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved → {out_path}]")


# -----------------------------------------------------------------------------
def plot_trajectories_2d(recs: list[dict], out_path):
    """Сетка 2D-траекторий поверх контуров f. Только n=2.

    По одному subplot на (функция × оптимизатор). Каждая траектория из J стартов
    рисуется тонкой линией, чтобы общий «характер» был виден."""
    recs2 = [r for r in recs if r["n"] == 2]
    if not recs2:
        print("[plot_trajectories_2d] no n=2 records, skipping")
        return
    groups = _group(recs2)
    funcs = sorted(set(k[0] for k in groups.keys()))
    optimizers = sorted(set(r["optimizer"] for r in recs2))

    nrows, ncols = len(funcs), len(optimizers)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows), squeeze=False
    )

    for i, fname in enumerate(funcs):
        # Базовая функция для контуров (для shifted/rotated тоже базовая)
        base_name = fname.split("_")[0]
        b = ALL_BENCHMARKS.get(base_name)
        if b is None:
            continue
        lo, hi = b.domain
        xg = np.linspace(lo, hi, 80)
        yg = np.linspace(lo, hi, 80)
        XX, YY = np.meshgrid(xg, yg)
        ZZ = np.array([[b.f(np.array([x, y])) for x in xg] for y in yg])

        for j, opt_name in enumerate(optimizers):
            ax = axs[i, j]
            runs = groups[(fname, 2)].get(opt_name, [])
            ax.contour(
                XX,
                YY,
                np.log(ZZ - ZZ.min() + 1e-3),
                levels=18,
                cmap="viridis",
                linewidths=0.5,
                alpha=0.7,
            )
            for r in runs:
                xs = np.array(r["xs"])
                ax.plot(
                    xs[:, 0], xs[:, 1], "-", linewidth=0.8, color="crimson", alpha=0.5
                )
                ax.scatter(
                    xs[0, 0], xs[0, 1], s=10, color="black", marker="s", zorder=3
                )
                ax.scatter(
                    xs[-1, 0], xs[-1, 1], s=14, color="red", marker="*", zorder=4
                )
            if i == 0:
                ax.set_title(opt_name, fontsize=9)
            if j == 0:
                ax.set_ylabel(fname, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Trajectories in 2D (black ◾ start, red ★ end)", fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved → {out_path}]")


# -----------------------------------------------------------------------------
def plot_metrics_heatmap(recs: list[dict], out_path, metric: str = "success_rate"):
    """Heatmap: строки — (function, n), столбцы — optimizer, цвет — метрика."""
    by_key = {}
    for r in recs:
        key = (r["function"], r["n"], r["optimizer"])
        by_key.setdefault(key, []).append(r)

    funcs_n = sorted(set((k[0], k[1]) for k in by_key.keys()))
    optimizers = sorted(set(k[2] for k in by_key.keys()))

    M = np.full((len(funcs_n), len(optimizers)), np.nan)
    annot = [[""] * len(optimizers) for _ in range(len(funcs_n))]
    for i, (fn, n) in enumerate(funcs_n):
        for j, opt in enumerate(optimizers):
            rs = by_key.get((fn, n, opt), [])
            if not rs:
                continue
            if metric == "success_rate":
                val = float(np.mean([1.0 if r["success"] else 0.0 for r in rs]))
                annot[i][j] = f"{val:.2f}"
            elif metric == "mean_final_gap":
                gaps = [r["final_gap"] for r in rs if np.isfinite(r["final_gap"])]
                val = float(np.mean(gaps)) if gaps else np.nan
                annot[i][j] = f"{val:.2e}" if np.isfinite(val) else "—"
            elif metric == "mean_best_gap":
                gaps = [r["best_gap"] for r in rs if np.isfinite(r["best_gap"])]
                val = float(np.mean(gaps)) if gaps else np.nan
                annot[i][j] = f"{val:.2e}" if np.isfinite(val) else "—"
            else:
                val = float(np.nanmean([r.get(metric, np.nan) for r in rs]))
                annot[i][j] = f"{val:.2f}"
            M[i, j] = val

    fig, ax = plt.subplots(
        figsize=(1.4 * len(optimizers) + 1.5, 0.5 * len(funcs_n) + 1)
    )
    cmap = "RdYlGn" if metric == "success_rate" else "RdYlGn_r"
    im = ax.imshow(
        M,
        cmap=cmap,
        aspect="auto",
        vmin=0 if metric == "success_rate" else None,
        vmax=1 if metric == "success_rate" else None,
    )
    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels(optimizers, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(funcs_n)))
    ax.set_yticklabels([f"{fn}, n={n}" for fn, n in funcs_n], fontsize=9)
    for i in range(len(funcs_n)):
        for j in range(len(optimizers)):
            ax.text(
                j, i, annot[i][j], ha="center", va="center", fontsize=8, color="black"
            )
    ax.set_title(f"{metric}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved → {out_path}]")


# -----------------------------------------------------------------------------
def plot_cosine_per_step(recs: list[dict], out_path, function: str, n: int = 2):
    """cos(шаг, -grad) по шагам для всех оптимизаторов на одной (функция, n)."""
    recs_f = [r for r in recs if r["function"] == function and r["n"] == n]
    if not recs_f:
        print(f"[plot_cosine_per_step] no records for {function}, n={n}")
        return
    groups = {}
    for r in recs_f:
        groups.setdefault(r["optimizer"], []).append(r)

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("tab10")
    for idx, (opt_name, runs) in enumerate(sorted(groups.items())):
        cosines = np.array([r["cosine_per_step"] for r in runs])  # (J, K)
        mean = np.nanmean(cosines, axis=0)
        p10 = np.nanpercentile(cosines, 10, axis=0)
        p90 = np.nanpercentile(cosines, 90, axis=0)
        c = cmap(idx % 10)
        ax.plot(mean, label=opt_name, color=c, linewidth=1.4)
        ax.fill_between(range(len(mean)), p10, p90, color=c, alpha=0.12)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\cos(\Delta x_k,\, -\nabla f(x_k))$")
    ax.set_title(f"Alignment with antigradient: {function}, n={n}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved → {out_path}]")
