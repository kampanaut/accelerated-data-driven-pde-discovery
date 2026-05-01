#!/usr/bin/env python3
"""Per-PDE per-mode feature-distribution table with resampling-based R^2.

Methodology:
  EACH (PDE, mode):
    EACH task:
      For iter in 1..N_iter:
        Sample one batch via task.get_support_query_split(query_size=N_points, ...)
        Compute multivariate OLS R^2 on this sample (per mixer).
      Per-mixer per-task summary: mean, std, min, max of the N_iter R^2 values.

    Across-task aggregation per mixer:
      4x4 grid: outer{mean,std,min,max} of inner{mean,std,min,max}.

Sample sizes:
  single-equation PDEs (Heat, NLHeat) -> 20000 points
  two-equation PDEs (BR, LO)          -> 40000 points (same points feed both mixers)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.task_loader import TASK_REGISTRY  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 42

CONFIGS = [
    {
        "label": "Heat",
        "pde_type": "heat",
        "test_dir": "data/datasets/heat_test-1",
        "train_dir": "data/datasets/heat_train-2",
        "modes": ["library"],
        "n_points": 20000,
    },
    {
        "label": "NLHeat",
        "pde_type": "nl_heat",
        "test_dir": "data/datasets/nl_heat_test-1",
        "train_dir": "data/datasets/nl_heat_train-2",
        "modes": ["precompose", "raw"],
        "n_points": 20000,
    },
    {
        "label": "BR",
        "pde_type": "br",
        "test_dir": "data/datasets/br_test-2",
        "train_dir": "data/datasets/br_train-2",
        "modes": ["library", "raw"],
        "n_points": 40000,
    },
    {
        "label": "LO",
        "pde_type": "lo",
        "test_dir": "data/datasets/lo_test-1",
        "train_dir": "data/datasets/lo_train-1",
        "modes": ["library", "raw"],
        "n_points": 40000,
    },
]


def r_squared(y: torch.Tensor, X: torch.Tensor) -> float:
    """OLS R^2 of y on X (with intercept). Uses CPU + pinv for numerical stability;
    torch.linalg.lstsq on CUDA NaNs out for some LO raw target distributions."""
    X_d = X.double().cpu()
    y_d = y.double().cpu()
    if y_d.ndim == 1:
        y_d = y_d.unsqueeze(1)
    X_aug = torch.cat([X_d, torch.ones(X_d.shape[0], 1, dtype=torch.float64)], dim=1)
    # Pseudoinverse is robust to rank-deficient or ill-conditioned matrices.
    sol = torch.linalg.pinv(X_aug) @ y_d
    y_pred = X_aug @ sol
    ss_res = float(((y_d - y_pred) ** 2).sum())
    ss_tot = float(((y_d - y_d.mean()) ** 2).sum())
    if not (ss_res == ss_res) or not (ss_tot == ss_tot):  # NaN check
        return float("nan")
    return 1.0 - (ss_res / max(ss_tot, 1e-30))


def collect_per_task_r2(
    pde_type: str, data_dir: Path, mode: str, n_points: int, n_iter: int,
    noise_level: float = 0.0,
):
    """For each task in directory: N_iter resamples, compute R^2 per mixer.

    Returns list-per-mixer of arrays of shape (n_tasks, n_iter).
    """
    files = sorted(data_dir.glob("*_fourier.npz"))
    if not files:
        raise SystemExit(f"no *_fourier.npz in {data_dir}")
    task_class = TASK_REGISTRY[pde_type]

    first = task_class(files[0], device=DEVICE, input_mode=mode)
    n_mixers = first.n_targets

    # r2_per_mixer[m][task_idx] = list of N_iter R^2 values
    r2_per_mixer = [[] for _ in range(n_mixers)]

    for t_idx, f in enumerate(files):
        task = task_class(f, device=DEVICE, input_mode=mode)
        per_iter_r2 = [[] for _ in range(n_mixers)]
        for it in range(n_iter):
            seed = BASE_SEED + t_idx * 10_000 + it
            # Per-iter seeded noise generator so noise injection is
            # reproducible and independent across resamples.
            noise_gen = None
            if noise_level > 0.0:
                noise_gen = torch.Generator(device=DEVICE)
                noise_gen.manual_seed(seed)
            try:
                (_, _), (qf_list, qt), _, _ = task.get_support_query_split(
                    K_shot=0, query_size=n_points,
                    k_seed=seed, snapshot_seed=seed,
                    noise_level=noise_level, noise_generator=noise_gen,
                )
            except Exception:
                # fall back if K_shot=0 disallowed
                (sf_list, st), (qf_list, qt), _, _ = task.get_support_query_split(
                    K_shot=1, query_size=n_points - 1,
                    k_seed=seed, snapshot_seed=seed,
                    noise_level=noise_level, noise_generator=noise_gen,
                )
                # combine support+query for the regression
                qf_list = [torch.cat([s, q], dim=0) for s, q in zip(sf_list, qf_list)]
                qt = torch.cat([st, qt], dim=0)
            for m in range(n_mixers):
                r2 = r_squared(qt[:, m], qf_list[m])
                per_iter_r2[m].append(r2)
        for m in range(n_mixers):
            r2_per_mixer[m].append(per_iter_r2[m])
        del task

    out = []
    for m in range(n_mixers):
        arr = np.array(r2_per_mixer[m])  # (n_tasks, n_iter)
        out.append(arr)
    return out, [f.stem for f in files]


def task_inner_stats(arr_iter: np.ndarray) -> dict:
    """Inner stats for one task's R^2 array of shape (n_iter,)."""
    return {
        "mean": float(arr_iter.mean()),
        "std": float(arr_iter.std()),
        "min": float(arr_iter.min()),
        "max": float(arr_iter.max()),
    }


def across_task_grid(per_task_inner: list[dict]) -> dict:
    """4x4 grid of outer{mean,std,min,max} of inner{mean,std,min,max}."""
    by_inner = {k: np.array([t[k] for t in per_task_inner]) for k in ["mean", "std", "min", "max"]}
    grid = {}
    for outer in ["mean", "std", "min", "max"]:
        for inner in ["mean", "std", "min", "max"]:
            arr = by_inner[inner]
            if outer == "mean":
                grid[f"{outer}_of_{inner}"] = float(arr.mean())
            elif outer == "std":
                grid[f"{outer}_of_{inner}"] = float(arr.std())
            elif outer == "min":
                grid[f"{outer}_of_{inner}"] = float(arr.min())
            elif outer == "max":
                grid[f"{outer}_of_{inner}"] = float(arr.max())
    return grid


def print_grid(label: str, grid: dict, n_tasks: int, n_iter: int):
    print(f"\n  {'inner':<10s}{'mean':>14s}{'std':>14s}{'min':>14s}{'max':>14s}")
    print("  " + "-" * 66)
    for outer in ["mean", "std", "min", "max"]:
        cells = [grid[f"{outer}_of_{inner}"] for inner in ["mean", "std", "min", "max"]]
        print(f"  {outer:<10s}" + "".join(f"{c:>14.6f}" for c in cells))
    print(f"\n  (n_tasks={n_tasks}, n_iter={n_iter})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--only", type=str, default="", help="comma-separated PDE labels filter")
    parser.add_argument(
        "--noise_level", type=float, default=0.0,
        help="Source-level Fourier-coefficient noise scale (matches eval pipeline).",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "train"],
        help="Which dataset split to evaluate feasibility R^2 on.",
    )
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None

    headlines = []  # for final compact summary

    for cfg in CONFIGS:
        if only and cfg["label"] not in only:
            continue
        data_dir = ROOT / cfg[f"{args.split}_dir"]
        if not data_dir.exists():
            print(f"\n[skip] {cfg['label']} ({args.split}): {data_dir} not found")
            continue
        for mode in cfg["modes"]:
            print(f"\n{'=' * 80}")
            print(f"  {cfg['label']}  |  split = {args.split}  |  mode = {mode}  |  N_points = {cfg['n_points']}  |  N_iter = {args.n_iter}  |  noise = {args.noise_level}")
            print(f"{'=' * 80}")
            try:
                arr_per_mixer, task_names = collect_per_task_r2(
                    cfg["pde_type"], data_dir, mode, cfg["n_points"], args.n_iter,
                    noise_level=args.noise_level,
                )
            except Exception as e:
                print(f"  [fail] {type(e).__name__}: {e}")
                continue
            n_tasks = len(task_names)
            for m_idx, arr in enumerate(arr_per_mixer):
                # arr: (n_tasks, n_iter)
                per_task_inner = [task_inner_stats(arr[t]) for t in range(n_tasks)]
                grid = across_task_grid(per_task_inner)
                mlab = ("u_t" if m_idx == 0 else "v_t") if len(arr_per_mixer) > 1 else "u_t"
                print(f"\n  --- mixer {m_idx} ({mlab}) ---")
                print_grid(f"{cfg['label']}/{mode}/m{m_idx}", grid, n_tasks, args.n_iter)
                headlines.append({
                    "pde": cfg["label"],
                    "mode": mode,
                    "mixer": mlab,
                    "n_tasks": n_tasks,
                    "n_iter": args.n_iter,
                    "grid": grid,
                })

    # Compact summary table — diagonal of the 4x4 grid (most-common interpretive entries)
    print(f"\n{'=' * 95}")
    print("  COMPACT SUMMARY — outer{mean,min,max} of inner{mean}, plus mean(std) (sampling noise)")
    print(f"{'=' * 95}")
    hdr = (
        f"  {'PDE':<8s}{'mode':<12s}{'mixer':<6s}"
        f"{'mean(mean)':>14s}{'std(mean)':>14s}{'min(mean)':>14s}{'max(mean)':>14s}{'mean(std)':>14s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for h in headlines:
        g = h["grid"]
        print(
            f"  {h['pde']:<8s}{h['mode']:<12s}{h['mixer']:<6s}"
            f"{g['mean_of_mean']:>14.6f}{g['std_of_mean']:>14.6f}"
            f"{g['min_of_mean']:>14.6f}{g['max_of_mean']:>14.6f}"
            f"{g['mean_of_std']:>14.6e}"
        )


if __name__ == "__main__":
    main()
