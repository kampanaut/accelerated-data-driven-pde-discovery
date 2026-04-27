#!/usr/bin/env python3
"""Coefficient recovery analysis: err, slope, r, wins, R² per noise level.

Usage:
    python scripts/compare_coefficient_recovery.py data/models/.../experiment_name
    python scripts/compare_coefficient_recovery.py data/models/.../exp1 data/models/.../exp2
    python scripts/compare_coefficient_recovery.py --config configs/some_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def analyze_experiment(exp_dir: Path) -> None:
    results_path = exp_dir / "evaluation" / "results.json"
    sample_dir = exp_dir / "evaluation" / "samples"

    if not results_path.exists():
        print(f"  SKIP: no results.json in {exp_dir}")
        return

    with open(results_path) as f:
        results = json.load(f)

    tasks = results["tasks"]
    n_tasks = len(tasks)

    # Discover coefficients and noise levels
    coeff_names: list[str] = []
    noise_levels: set[float] = set()
    for tdata in tasks.values():
        for combo in tdata["combos"]:
            noise_levels.add(combo["noise"])
            for method_name in ["maml", "baseline"]:
                if method_name not in combo:
                    continue
                cr = combo[method_name].get("coefficient_recovery", {})
                coeffs = cr.get("coefficients", {})
                for cname in coeffs:
                    if cname not in coeff_names:
                        coeff_names.append(cname)

    noise_sorted = sorted(noise_levels)

    print(f"\n{'=' * 110}")
    print(f"  {exp_dir.name}")
    print(f"  {n_tasks} tasks, coefficients: {coeff_names}")
    print(f"{'=' * 110}")

    for cname in coeff_names:
        true_vals = [
            tdata["coefficients"][cname]
            for tdata in tasks.values()
            if cname in tdata.get("coefficients", {})
        ]
        if true_vals:
            arr = np.asarray(true_vals, dtype=float)
            header_tag = (
                f"{cname} [mean={arr.mean():.6f}, std={arr.std():.6f}, "
                f"min={arr.min():.6f}, max={arr.max():.6f}, n={len(arr)}]"
            )
        else:
            header_tag = cname
        print(f"\n--- {header_tag} ---")
        header = (
            f"{'noise':>6s} {'method':>8s}  {'err mean±std':>16s}  "
            f"{'min':>7s}  {'max':>7s}  {'slope':>7s}  {'r':>6s}  "
            f"{'wins':>7s}  {'R² mean±std':>14s}  {'R²min':>7s}  {'R²max':>7s}"
        )
        print(header)
        print("-" * len(header))

        for noise_target in noise_sorted:
            method_data: dict[str, dict] = {}

            for method_name in ["maml", "baseline"]:
                errs: list[float] = []
                recs: list[float] = []
                trues: list[float] = []
                r2s: list[float] = []

                for tname in sorted(tasks.keys()):
                    tdata = tasks[tname]
                    true_val = tdata["coefficients"].get(cname)
                    if true_val is None:
                        continue

                    for combo in tdata["combos"]:
                        if abs(combo["noise"] - noise_target) > 1e-6:
                            continue
                        if method_name not in combo:
                            continue
                        cr = combo[method_name].get("coefficient_recovery", {})
                        coeffs = cr.get("coefficients", {})
                        if cname not in coeffs:
                            continue
                        last = coeffs[cname]["per_step"][-1]
                        errs.append(last["pct_error"])
                        recs.append(last["cross_path_mean"])
                        trues.append(true_val)

                    # R² from NPZ
                    if sample_dir.exists():
                        npz_path = sample_dir / f"{tname}.npz"
                        if npz_path.exists():
                            npz = np.load(npz_path, allow_pickle=True)
                            noise_str = f"{noise_target:.2f}"
                            for key in npz.keys():
                                if (
                                    f"noise_{noise_str}/{method_name}/raw_values/"
                                    in key
                                ):
                                    reg_key = key.replace(
                                        "raw_values", "regressors"
                                    )
                                    if reg_key in npz:
                                        v = npz[key][-1]
                                        r = npz[reg_key][-1]
                                        y = v * r
                                        x = r
                                        sl = float(
                                            (x * y).sum()
                                            / ((x * x).sum() + 1e-30)
                                        )
                                        ss_res = float(
                                            ((y - sl * x) ** 2).sum()
                                        )
                                        ss_tot = float(
                                            ((y - y.mean()) ** 2).sum()
                                        )
                                        r2 = (
                                            1.0 - (ss_res / ss_tot)
                                            if ss_tot > 0
                                            else 0.0
                                        )
                                        r2s.append(r2)
                                    break

                method_data[method_name] = {
                    "errs": np.array(errs),
                    "recs": np.array(recs),
                    "trues": np.array(trues),
                    "r2s": np.array(r2s),
                }

            maml_d = method_data.get("maml", {})
            bl_d = method_data.get("baseline", {})
            maml_errs = maml_d.get("errs", np.array([]))
            maml_r2s = maml_d.get("r2s", np.array([]))
            bl_errs = bl_d.get("errs", np.array([]))
            bl_r2s = bl_d.get("r2s", np.array([]))

            # Product score: err% × (1 - R²). Lower = better.
            # Rewards both accurate slope and tight scatter.
            can_score = (
                len(maml_errs) == len(bl_errs)
                and len(maml_r2s) == len(bl_r2s)
                and len(maml_errs) == len(maml_r2s)
                and len(maml_errs) > 0
            )
            if can_score:
                maml_score = maml_errs * (1.0 - np.clip(maml_r2s, -1, 1))
                bl_score = bl_errs * (1.0 - np.clip(bl_r2s, -1, 1))

            for method_name in ["maml", "baseline"]:
                d = method_data.get(method_name, {})
                errs = d.get("errs", np.array([]))
                recs = d.get("recs", np.array([]))
                trues = d.get("trues", np.array([]))
                r2s = d.get("r2s", np.array([]))

                if len(errs) == 0:
                    continue

                # Cross-task slope and r
                if len(trues) > 1:
                    slope = float(np.polyfit(trues, recs, 1)[0])
                    corr = float(np.corrcoef(trues, recs)[0, 1])
                else:
                    slope, corr = float("nan"), float("nan")

                # Wins (product score)
                if can_score:
                    if method_name == "maml":
                        wins = int((maml_score < bl_score).sum())
                    else:
                        wins = int((bl_score < maml_score).sum())
                else:
                    wins = 0
                total = len(errs)

                # R² stats
                if len(r2s) > 0:
                    r2_str = f"{r2s.mean():.3f}±{r2s.std():.3f}"
                    r2_min = f"{r2s.min():.3f}"
                    r2_max = f"{r2s.max():.3f}"
                else:
                    r2_str, r2_min, r2_max = "", "", ""

                label = "MAML" if method_name == "maml" else "BL"
                print(
                    f"{noise_target:6.2f} {label:>8s}  "
                    f"{errs.mean():6.2f}±{errs.std():6.2f}%  "
                    f"{errs.min():6.2f}%  {errs.max():6.2f}%  "
                    f"{slope:7.3f}  {corr:6.3f}  "
                    f"{wins:3d}/{total:<3d}  "
                    f"{r2_str:>14s}  {r2_min:>7s}  {r2_max:>7s}"
                )


def plot_training_loss(exp_dirs: list[Path], out_path: Path) -> None:
    """Training loss overlay for all experiments."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    for exp_dir in exp_dirs:
        history_path = exp_dir / "training" / "history.json"
        if not history_path.exists():
            continue
        with open(history_path) as f:
            h = json.load(f)

        losses = np.array(h["train_loss"])
        iters = np.array(h.get("iteration", list(range(len(losses)))))
        label = exp_dir.name

        # Rolling mean (window=50)
        window = min(50, max(1, len(losses) // 5))
        if window > 1:
            kernel = np.ones(window) / window
            smooth = np.convolve(losses, kernel, mode="valid")
            smooth_iters = iters[: len(smooth)]
            (line,) = ax.plot(smooth_iters, smooth, label=label, linewidth=2)
        else:
            (line,) = ax.plot(iters, losses, label=label, linewidth=2)

        # Raw data faintly behind
        ax.plot(iters, losses, alpha=0.15, linewidth=0.5, color=line.get_color())

        # Per-mixer breakdown if available
        mixers = h.get("mixers", {})
        for mname, mdata in mixers.items():
            color = line.get_color()

            # Main nMSE (dashed)
            mse = mdata.get("mse_main", [])
            if mse and window > 1 and len(mse) > window:
                mse_smooth = np.convolve(np.array(mse), kernel, mode="valid")
                ax.plot(
                    iters[: len(mse_smooth)], mse_smooth,
                    linestyle="--", linewidth=1, color=color, alpha=0.6,
                    label=f"{label} m[{mname}] mse",
                )

            # Aux losses (dotted, one per coefficient)
            aux = mdata.get("aux", {})
            if isinstance(aux, dict):
                for aux_name, aux_vals in aux.items():
                    if aux_vals and window > 1 and len(aux_vals) > window:
                        aux_smooth = np.convolve(np.array(aux_vals), kernel, mode="valid")
                        ax.plot(
                            iters[: len(aux_smooth)], aux_smooth,
                            linestyle=":", linewidth=1, color=color, alpha=0.5,
                            label=f"{label} m[{mname}] aux:{aux_name}",
                        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("Meta-Training Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nTraining loss plot saved to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "exp_dirs",
        nargs="*",
        help="Experiment directories (containing evaluation/results.json)",
    )
    parser.add_argument(
        "--config", type=str, help="Config YAML (resolves experiment dir)"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip training loss plot"
    )
    args = parser.parse_args()

    exp_dirs: list[Path] = []

    if args.config:
        from src.config import ExperimentConfig

        cfg = ExperimentConfig.from_yaml(args.config)
        exp_dirs.append(cfg.exp_dir)

    for d in args.exp_dirs:
        exp_dirs.append(Path(d))

    if not exp_dirs:
        print(
            "Usage: python scripts/compare_coefficient_recovery.py "
            "<exp_dir> [exp_dir ...] [--config config.yaml]"
        )
        sys.exit(1)

    for exp_dir in exp_dirs:
        analyze_experiment(exp_dir)

    if not args.no_plot:
        out_path = Path("figures") / "training_loss_comparison.png"
        out_path.parent.mkdir(exist_ok=True)
        plot_training_loss(exp_dirs, out_path)


if __name__ == "__main__":
    main()
