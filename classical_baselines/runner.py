"""
runner.py — главный CLI-скрипт.

Команды:
    tune-lr        Подбор lr для классических методов на сетке (function × start).
    run-classical  Прогон классических методов на полном протоколе:
                   все функции × n × J стартов × все методы.
    run-llm        Один интерактивный запуск LLM на (функция, n, старт).
    compare        Свести лог LLM + классики в единые таблицы/графики.

Все логи и результаты пишутся в --outdir (по умолчанию ./logs/).

Примеры запуска — смотри в README.md.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from benchmarks import ALL_BENCHMARKS
from classical import ALL_OPTIMIZERS, grid_search_lr
from llm_interactive import llm_optimize_interactive, llm_optimize_replay
from metrics import all_metrics_for_trajectory


# Стандартный набор гиперпараметров за рамками lr
EXTRA_HP = {
    "gd": {},
    "heavy_ball": {"beta": 0.9},
    "nesterov": {"beta": 0.9},
    "adam": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "adamw": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01},
    "bfgs": {},
}


def make_starts(
    n: int,
    J: int,
    domain: tuple[float, float],
    seed: int = 42,
    shift_from_origin: float = 1.0,
) -> np.ndarray:
    """J случайных стартов в domain, сдвинутых от 0 для борьбы с origin-bias."""
    rng = np.random.default_rng(seed)
    lo, hi = domain
    starts = rng.uniform(lo, hi, size=(J, n))
    # отталкиваем точки от нуля: если близко к 0, прибавляем shift
    norms = np.linalg.norm(starts, axis=1, keepdims=True)
    too_close = norms < shift_from_origin
    starts = np.where(
        too_close, starts + shift_from_origin * np.sign(starts + 1e-12), starts
    )
    return starts


# -----------------------------------------------------------------------------
# Subcommand: tune-lr
# -----------------------------------------------------------------------------
def cmd_tune_lr(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "tuned_lrs.json"

    results = {}
    for fname in args.functions:
        b = ALL_BENCHMARKS[fname]
        results[fname] = {}
        for n in args.dims:
            print(f"\n=== {fname}, n={n} ===")
            starts = make_starts(n, args.J, b.domain, seed=args.seed)
            # для подбора lr используем среднее по J стартам
            for opt in args.optimizers:
                if opt == "bfgs":
                    continue  # BFGS не нуждается в подборе lr
                finals = []
                lrs_per_start = []
                for j, x0 in enumerate(starts):
                    lr, final = grid_search_lr(
                        opt,
                        b.f,
                        b.grad,
                        x0,
                        args.K,
                        extra_hparams=EXTRA_HP[opt],
                    )
                    finals.append(final)
                    lrs_per_start.append(lr)
                # берём медианный lr (устойчиво к выбросам)
                best_lr = float(np.median(lrs_per_start))
                mean_final = float(np.mean([x for x in finals if np.isfinite(x)]))
                results[fname].setdefault(str(n), {})[opt] = {
                    "lr": best_lr,
                    "mean_final": mean_final,
                    "per_start_lrs": lrs_per_start,
                }
                print(
                    f"  {opt:12} median_lr={best_lr:.4f}  mean_final={mean_final:.4e}"
                )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved tuned lrs to {out_path}]")


# -----------------------------------------------------------------------------
# Subcommand: run-classical
# -----------------------------------------------------------------------------
def cmd_run_classical(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ["BLOCK2_OUTDIR"] = str(outdir)

    # Загружаем lrs (либо tuned, либо дефолт)
    tuned = {}
    if args.tuned_lrs:
        with open(args.tuned_lrs) as f:
            tuned = json.load(f)

    all_results = []
    for fname in args.functions:
        b = ALL_BENCHMARKS[fname]
        for n in args.dims:
            starts = make_starts(n, args.J, b.domain, seed=args.seed)
            x_star = b.x_star_fn(n)
            for opt_name in args.optimizers:
                lr = _get_lr(tuned, fname, n, opt_name, default=args.default_lr)
                for j, x0 in enumerate(starts):
                    opt = ALL_OPTIMIZERS[opt_name]
                    kwargs = dict(EXTRA_HP[opt_name])
                    if opt_name != "bfgs":
                        kwargs["lr"] = lr
                    t0 = time.time()
                    with np.errstate(over="ignore", invalid="ignore"):
                        xs, fs = opt(b.f, b.grad, x0, args.K, **kwargs)
                    elapsed = time.time() - t0
                    m = all_metrics_for_trajectory(
                        xs, fs, b.grad, b.f_star, x_star, eps=args.eps
                    )
                    rec = {
                        "function": fname,
                        "n": n,
                        "optimizer": opt_name,
                        "lr": lr if opt_name != "bfgs" else None,
                        "start_idx": j,
                        "x0": x0.tolist(),
                        "xs": xs.tolist(),
                        "fs": fs.tolist(),
                        "elapsed_s": elapsed,
                        **m,
                    }
                    all_results.append(rec)
            print(f"  {fname:12} n={n} {opt_name:12} lr={lr} done")
    out = outdir / "classical_runs.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved {len(all_results)} runs to {out}]")

    # Сразу выведем красивую таблицу
    _print_summary_table(all_results, args.eps)


def _get_lr(tuned: dict, fname: str, n: int, opt_name: str, default: float) -> float:
    try:
        return float(tuned[fname][str(n)][opt_name]["lr"])
    except KeyError:
        return default


# -----------------------------------------------------------------------------
# Subcommand: run-llm
# -----------------------------------------------------------------------------
def cmd_run_llm(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    b = ALL_BENCHMARKS[args.function]
    n = args.n
    starts = make_starts(n, args.J, b.domain, seed=args.seed)
    if args.start_idx >= args.J:
        sys.exit(f"start_idx {args.start_idx} >= J {args.J}")
    x0 = starts[args.start_idx]

    log_path = (
        outdir / f"llm_{args.model}_{args.function}_n{n}_start{args.start_idx}.json"
    )

    print(
        f"Run LLM '{args.model}' on {args.function} (n={n}), start #{args.start_idx} = {x0}"
    )
    print(f"K = {args.K} steps,  log → {log_path}")

    if args.replay:
        xs, fs = llm_optimize_replay(b.f, b.grad, x0, args.K, replay_log=args.replay)
    else:
        xs, fs = llm_optimize_interactive(
            b.f,
            b.grad,
            x0,
            args.K,
            history_len=args.history_len,
            log_path=log_path,
        )

    x_star = b.x_star_fn(n)
    m = all_metrics_for_trajectory(xs, fs, b.grad, b.f_star, x_star, eps=args.eps)
    # Запишем итоговую запись для последующего сравнения
    rec = {
        "function": args.function,
        "n": n,
        "optimizer": f"llm_{args.model}",
        "lr": None,
        "start_idx": args.start_idx,
        "x0": x0.tolist(),
        "xs": xs.tolist(),
        "fs": fs.tolist(),
        **m,
    }
    out = (
        outdir
        / f"llm_run_{args.model}_{args.function}_n{n}_start{args.start_idx}_summary.json"
    )
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\n[summary saved to {out}]")
    print(f"  final_gap = {m['final_gap']:.4e}")
    print(f"  best_gap  = {m['best_gap']:.4e}")
    print(f"  T_eps     = {m['T_eps']}")
    print(f"  mean cosine antigrad = {m['mean_cosine_antigrad']:.3f}")
    print(f"  mean step consistency = {m['mean_step_consistency']:.3f}")


# -----------------------------------------------------------------------------
# Subcommand: compare — свести всё, посчитать таблицы, нарисовать графики
# -----------------------------------------------------------------------------
def cmd_compare(args):
    """Свести classical_runs.json + все llm-*_summary.json в одно."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ["BLOCK2_OUTDIR"] = str(outdir)
    all_recs = []
    cl_path = outdir / "classical_runs.json"
    if cl_path.exists():
        with open(cl_path) as f:
            all_recs.extend(json.load(f))
    # подбираем LLM-summary-файлы
    for p in outdir.glob("llm_run_*_summary.json"):
        with open(p) as f:
            all_recs.append(json.load(f))

    if not all_recs:
        sys.exit(f"No records found in {outdir}")

    print(f"Loaded {len(all_recs)} run records.")

    # сохраняем сводный файл
    combined = outdir / "combined_runs.json"
    with open(combined, "w") as f:
        json.dump(all_recs, f, indent=2)
    print(f"[combined → {combined}]")

    # Печатаем сводную таблицу
    _print_summary_table(all_recs, args.eps)

    # Строим графики
    if not args.no_plots:
        from plots import (
            plot_convergence_grid,
            plot_trajectories_2d,
            plot_metrics_heatmap,
        )

        plot_convergence_grid(all_recs, outdir / "fig_convergence_grid.png")
        plot_trajectories_2d(all_recs, outdir / "fig_trajectories_2d.png")
        plot_metrics_heatmap(all_recs, outdir / "fig_metrics_heatmap.png")


def _print_summary_table(recs, eps):
    """Печатает агрегированную таблицу: (function × n × optimizer) → метрики."""
    by_key = {}
    for r in recs:
        key = (r["function"], r["n"], r["optimizer"])
        by_key.setdefault(key, []).append(r)

    print(
        f"\n{'function':12} {'n':>3} {'optimizer':14} "
        f"{'mean_gap':>11} {'best_gap':>11} "
        f"{'succ_rate':>9} {'mean_T':>7} {'mean_cos':>9}"
    )
    print("-" * 95)
    rows_for_csv = []
    for key in sorted(by_key.keys()):
        rs = by_key[key]
        mean_gap = float(
            np.mean([r["final_gap"] for r in rs if np.isfinite(r["final_gap"])])
        )
        best_gap = float(np.min([r["best_gap"] for r in rs]))
        succ = float(np.mean([1.0 if r["success"] else 0.0 for r in rs]))
        T_vals = [r["T_eps"] for r in rs if r["T_eps"] is not None]
        mean_T = float(np.mean(T_vals)) if T_vals else float("nan")
        mean_cos = float(np.nanmean([r["mean_cosine_antigrad"] for r in rs]))
        print(
            f"{key[0]:12} {key[1]:>3} {key[2]:14} "
            f"{mean_gap:>11.3e} {best_gap:>11.3e} "
            f"{succ:>9.2f} {mean_T:>7.1f} {mean_cos:>9.3f}"
        )
        rows_for_csv.append(
            {
                "function": key[0],
                "n": key[1],
                "optimizer": key[2],
                "mean_gap": mean_gap,
                "best_gap": best_gap,
                "success_rate": succ,
                "mean_T_eps": mean_T,
                "mean_cosine_antigrad": mean_cos,
            }
        )

    # сохраняем агрегированную таблицу в CSV рядом с combined
    import csv

    # outdir передаётся через args, но здесь его нет — определим через первый rec
    # или через переменную окружения, которую установит вызывающий
    import os

    csv_path = Path(os.environ.get("BLOCK2_OUTDIR", "logs")) / "summary_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        if rows_for_csv:
            w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_for_csv)
    print(f"\n[summary table → {csv_path}]")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Block 2 runner")
    sp = p.add_subparsers(dest="cmd", required=True)

    # tune-lr
    p_tl = sp.add_parser("tune-lr", help="Подобрать lr для классических методов")
    p_tl.add_argument("--functions", nargs="+", default=list(ALL_BENCHMARKS.keys()))
    p_tl.add_argument("--dims", nargs="+", type=int, default=[2, 5])
    p_tl.add_argument(
        "--optimizers",
        nargs="+",
        default=["gd", "heavy_ball", "nesterov", "adam", "adamw"],
    )
    p_tl.add_argument("--J", type=int, default=10)
    p_tl.add_argument("--K", type=int, default=50)
    p_tl.add_argument("--seed", type=int, default=42)
    p_tl.add_argument("--outdir", default="logs")
    p_tl.set_defaults(func=cmd_tune_lr)

    # run-classical
    p_rc = sp.add_parser("run-classical", help="Прогон классических методов")
    p_rc.add_argument("--functions", nargs="+", default=list(ALL_BENCHMARKS.keys()))
    p_rc.add_argument("--dims", nargs="+", type=int, default=[2, 5])
    p_rc.add_argument("--optimizers", nargs="+", default=list(ALL_OPTIMIZERS.keys()))
    p_rc.add_argument("--J", type=int, default=10)
    p_rc.add_argument("--K", type=int, default=50)
    p_rc.add_argument("--seed", type=int, default=42)
    p_rc.add_argument("--eps", type=float, default=1e-3)
    p_rc.add_argument(
        "--tuned-lrs",
        default="logs/tuned_lrs.json",
        help="JSON с оптимальными lr (см. tune-lr)",
    )
    p_rc.add_argument("--default-lr", type=float, default=0.01)
    p_rc.add_argument("--outdir", default="logs")
    p_rc.set_defaults(func=cmd_run_classical)

    # run-llm
    p_rl = sp.add_parser("run-llm", help="Один интерактивный запуск LLM")
    p_rl.add_argument(
        "--model", required=True, help="Имя модели (claude/gpt/gemini/deepseek)"
    )
    p_rl.add_argument("--function", required=True, choices=list(ALL_BENCHMARKS.keys()))
    p_rl.add_argument("--n", type=int, default=2)
    p_rl.add_argument("--J", type=int, default=10)
    p_rl.add_argument("--start-idx", type=int, default=0)
    p_rl.add_argument("--K", type=int, default=30)
    p_rl.add_argument("--seed", type=int, default=42)
    p_rl.add_argument("--eps", type=float, default=1e-3)
    p_rl.add_argument("--history-len", type=int, default=5)
    p_rl.add_argument(
        "--replay",
        default=None,
        help="Воспроизвести из существующего log-json (без LLM)",
    )
    p_rl.add_argument("--outdir", default="logs")
    p_rl.set_defaults(func=cmd_run_llm)

    # compare
    p_co = sp.add_parser("compare", help="Свести всё и построить таблицы/графики")
    p_co.add_argument("--outdir", default="logs")
    p_co.add_argument("--eps", type=float, default=1e-3)
    p_co.add_argument("--no-plots", action="store_true")
    p_co.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
