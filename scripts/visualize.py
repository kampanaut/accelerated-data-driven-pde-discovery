#!/usr/bin/env python3
"""
Generate visualizations from MAML evaluation results.

This script:
1. Loads results.json from evaluation
2. Computes metrics from stored loss curves
3. Generates all 5 graph types (per-task and aggregated)

Graph types:
- Graph 5: Convergence plots (Loss vs Steps)
- Graph 3: Speedup heatmaps (K × Noise)
- Graph 4: Loss ratio heatmaps (K × Noise)
- Graph 1: Noise robustness curves
- Graph 2: Sample efficiency curves

Usage:
    python scripts/visualize.py --config configs/experiment.yaml
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

from numpy.typing import NDArray
import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    compute_comparison_metrics,
    ComparisonMetrics,
    steps_to_plateau,
)
from src.evaluation.graphs import (
    plot_convergence,
    plot_train_holdout_convergence,
    plot_speedup_heatmap,
    plot_loss_ratio_heatmap,
    plot_noise_robustness,
    plot_sample_efficiency,
    # Graph 7-10: Jacobian analysis graphs
    plot_jacobian_histogram,
    plot_coefficient_heatmap,
    plot_coefficient_vs_k,
    plot_coefficient_vs_noise,
)


def load_config(config_path: Path) -> dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_results(results_path: Path) -> dict:
    """Load evaluation results (metadata only)."""
    with open(results_path, "r") as f:
        return json.load(f)


def load_samples(samples_dir: Path, task_name: str) -> Dict[str, np.ndarray]:
    """
    Load per-combo arrays from NPZ file for a task.

    Args:
        samples_dir: Directory containing .npz sample files
        task_name: Name of the task

    Returns:
        Dict mapping combo_key/array_name to ndarray, or empty dict if not found
    """
    npz_path = samples_dir / f"{task_name}.npz"
    if not npz_path.exists():
        return {}
    return dict(np.load(npz_path))


def _combo_worse_suffix(task_data: dict, combo_key: str) -> str:
    """Build WORSE suffix for a specific (K, noise) combo's image filename."""
    worse = task_data.get(f"worse_{combo_key}", {})
    flags = []
    if worse.get("loss", False):
        flags.append("LOSS")
    worse_coeffs = worse.get("coeff", [])
    if worse_coeffs:
        flags.append(f"COEFF:{','.join(worse_coeffs)}")
    if flags:
        return f"_WORSE[{','.join(flags)}]"
    return ""


def load_results_with_samples(results_path: Path) -> dict:
    """
    Load evaluation results with per-combo arrays from NPZ files.

    Args:
        results_path: Path to results.json

    Returns:
        Results dict with task_data['samples'][combo_key] populated from NPZ
    """
    results = load_results(results_path)

    samples_dir = results_path.parent / "samples"

    if not samples_dir.exists():
        raise ValueError("`samples_dir` directory path does not exist.")

    # Each .npz file about to be loaded by `load_samples()` is a task. A task
    # contains combos (K × noise level), ranging from few-shot clean data to
    # large-sample noisy data.
    for task_name, task_data in results["tasks"].items():
        raw = load_samples(samples_dir, task_name)
        if not raw:
            continue

        task_data["samples"] = {}

        # Discover combo_keys from NPZ key prefixes
        combo_keys = set()
        for key in raw.keys():
            combo_keys.add(key.rsplit("/")[0])

        # Extract coefficient names from task-level specs
        coeff_names = [s["name"] for s in task_data.get("coefficient_specs", [])]

        for combo_key in combo_keys:
            combo_data: Dict[str, Any] = {
                "maml_losses": raw.get(f"{combo_key}/maml_train_losses"),
                "baseline_losses": raw.get(f"{combo_key}/baseline_train_losses"),
                "maml_holdout_losses": raw.get(f"{combo_key}/maml_holdout_losses"),
                "baseline_holdout_losses": raw.get(
                    f"{combo_key}/baseline_holdout_losses"
                ),
                "maml_pred_errors": raw.get(f"{combo_key}/maml/pred_errors"),
                "baseline_pred_errors": raw.get(f"{combo_key}/baseline/pred_errors"),
            }
            for name in coeff_names:
                combo_data[f"maml_{name}"] = raw.get(f"{combo_key}/maml/{name}")
                combo_data[f"baseline_{name}"] = raw.get(f"{combo_key}/baseline/{name}")
                combo_data[f"maml_{name}_true"] = raw.get(
                    f"{combo_key}/maml/{name}_true"
                )
                combo_data[f"baseline_{name}_true"] = raw.get(
                    f"{combo_key}/baseline/{name}_true"
                )

            task_data["samples"][combo_key] = combo_data

            # Loss arrays to lists (metrics code expects lists)
            for k, v in combo_data.items():
                if "_losses" in k and v is not None:
                    combo_data[k] = v.tolist()

    return results


def compute_all_metrics(
    results: dict,
    fixed_steps: NDArray[np.integer[Any]],
    deriv_threshold: float = 1e-7,
) -> Dict[str, Dict[str, ComparisonMetrics]]:
    """
    Compute metrics for all (task, K, noise) combinations.

    Args:
        results: Loaded results data (with samples)
        threshold: L* threshold for legacy convergence detection
        fixed_steps: Steps at which to record loss values
        deriv_threshold: Maximum |derivative| for plateau detection

    Returns:
        Dict mapping task_name -> combo_key -> ComparisonMetrics
    """
    all_metrics = {}

    for task_name, task_data in results["tasks"].items():
        all_metrics[task_name] = {}

        if "samples" not in task_data:
            continue

        for combo_key, combo_data in task_data["samples"].items():
            maml_losses = combo_data.get("maml_losses")
            baseline_losses = combo_data.get("baseline_losses")

            if maml_losses is None or baseline_losses is None:
                continue

            # Get holdout losses if available
            maml_holdout = combo_data.get("maml_holdout_losses")
            baseline_holdout = combo_data.get("baseline_holdout_losses")

            metrics = compute_comparison_metrics(
                maml_losses=maml_losses,
                baseline_losses=baseline_losses,
                fixed_steps=fixed_steps,
                deriv_threshold=deriv_threshold,
                maml_holdout_losses=maml_holdout,
                baseline_holdout_losses=baseline_holdout,
            )
            all_metrics[task_name][combo_key] = metrics

    return all_metrics


def generate_per_task_figures(
    results: dict,
    all_metrics: Dict[str, Dict[str, ComparisonMetrics]],
    k_values: NDArray[np.integer[Any]],
    fixed_steps: NDArray[np.integer[Any]],
    noise_levels: NDArray[np.floating[Any]],
    output_dir: Path,
    dpi: int,
    deriv_threshold: float = 1e-7,
    holdout_size: int = 1000,
) -> None:
    """Generate all per-task figures."""

    for task_name, task_data in results["tasks"].items():
        worse_flags = []
        if task_data.get("loss_maml_worse", False):
            worse_flags.append("LOSS")
        worse_coeffs = task_data.get("coeff_maml_worse", [])
        if worse_coeffs:
            worse_flags.append(f"COEFF:{','.join(worse_coeffs)}")
        suffix = f"_WORSE[{','.join(worse_flags)}]" if worse_flags else ""

        task_dir = output_dir / "per_task" / f"{task_name}{suffix}"
        task_dir.mkdir(parents=True, exist_ok=True)

        task_metrics = all_metrics.get(task_name, {})
        task_samples = task_data["samples"]

        # Build coefficient grouping from task-level specs
        specs = task_data.get("coefficient_specs", [])
        coeff_group: Dict[str, List[Dict[str, Any]]] = {}
        for s in specs:
            coeff_group.setdefault(s["coeff_name"], []).append(s)

        print(f"  Generating figures for {task_name}...")

        # ---------------------------------------------------------------------
        # Graph 5: Convergence plots (one per K × noise)
        # ---------------------------------------------------------------------
        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"
                combo_data = task_samples.get(combo_key, {})

                maml_losses = combo_data.get("maml_losses")
                baseline_losses = combo_data.get("baseline_losses")

                if maml_losses is None or baseline_losses is None:
                    continue
                else:
                    maml_losses = np.array(maml_losses)
                    baseline_losses = np.array(baseline_losses)

                fig = plot_convergence(
                    maml_losses=maml_losses,
                    baseline_losses=baseline_losses,
                    title=f"{task_name}: K={k}, noise={noise:.0%}",
                    save_path=task_dir
                    / f"convergence_k{k}_noise{noise:.2f}{_combo_worse_suffix(task_data, combo_key)}.png",
                    dpi=dpi,
                    deriv_threshold=deriv_threshold,
                    k_shot=k,
                )
                plt.close(fig)

                # Train vs Holdout comparison (if holdout data available)
                maml_holdout = np.array(combo_data.get("maml_holdout_losses"))
                baseline_holdout = np.array(combo_data.get("baseline_holdout_losses"))
                if maml_holdout is not None and baseline_holdout is not None:
                    fig = plot_train_holdout_convergence(
                        maml_train=maml_losses,
                        maml_holdout=maml_holdout,
                        baseline_train=baseline_losses,
                        baseline_holdout=baseline_holdout,
                        title=f"{task_name}: K={k}, noise={noise:.0%} (Train vs Holdout)",
                        save_path=task_dir
                        / f"train_holdout_k{k}_noise{noise:.2f}{_combo_worse_suffix(task_data, combo_key)}.png",
                        dpi=dpi,
                        k_shot=k,
                        holdout_size=holdout_size,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 3: Speedup heatmap
        # ---------------------------------------------------------------------
        speedups = np.full((len(noise_levels), len(k_values)), np.nan)
        maml_losses_grid = np.full((len(noise_levels), len(k_values)), np.nan)
        baseline_losses_grid = np.full((len(noise_levels), len(k_values)), np.nan)

        found_count = 0
        for i, noise in enumerate(noise_levels):
            for j, k in enumerate(k_values):
                combo_key = f"k_{k}_noise_{noise:.2f}"
                metrics = task_metrics.get(combo_key)
                if metrics is not None:
                    found_count += 1
                    speedups[i, j] = metrics.speedup
                    maml_losses_grid[i, j] = metrics.maml_plateau_loss
                    baseline_losses_grid[i, j] = metrics.baseline_plateau_loss

        print(speedups)

        fig = plot_speedup_heatmap(
            speedups=speedups,
            k_values=k_values,
            noise_levels=noise_levels,
            title=f"{task_name}: Speedup Ratio",
            save_path=task_dir / "speedup_heatmap.png",
            dpi=dpi,
            maml_losses=maml_losses_grid,
            baseline_losses=baseline_losses_grid,
        )
        plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 4: Loss ratio heatmap (for first fixed step)
        # ---------------------------------------------------------------------
        for p in fixed_steps:
            ratios = np.full((len(noise_levels), len(k_values)), np.nan)

            for i, noise in enumerate(noise_levels):
                for j, k in enumerate(k_values):
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    metrics = task_metrics.get(combo_key)
                    if metrics is not None and p in metrics.loss_ratios:
                        ratios[i, j] = metrics.loss_ratios[p]

            fig = plot_loss_ratio_heatmap(
                ratios=ratios,
                k_values=k_values,
                noise_levels=noise_levels,
                fixed_step=p,
                title=f"{task_name}: Loss Ratio @ step {p}",
                save_path=task_dir / f"loss_ratio_heatmap_step{p}.png",
                dpi=dpi,
            )
            plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 1: Noise robustness curves (one per K)
        # ---------------------------------------------------------------------
        for k in k_values:
            maml_steps_list = []
            baseline_steps_list = []
            valid_noise = []

            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"
                metrics = task_metrics.get(combo_key)
                if metrics is None:
                    continue

                # Use steps_to_plateau (primary metric)
                maml_steps_list.append(metrics.maml_steps_to_plateau)
                baseline_steps_list.append(metrics.baseline_steps_to_plateau)
                valid_noise.append(noise)

            if len(valid_noise) >= 2:
                fig = plot_noise_robustness(
                    noise_levels=np.array(valid_noise),
                    maml_steps=np.array(maml_steps_list),
                    baseline_steps=np.array(baseline_steps_list),
                    title=f"{task_name}: Noise Robustness (K={k})",
                    save_path=task_dir / f"noise_robustness_k{k}.png",
                    dpi=dpi,
                )
                plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 2: Sample efficiency curves (one per noise level)
        # ---------------------------------------------------------------------
        for p in fixed_steps[:1]:  # Just first fixed step
            for noise in noise_levels:
                maml_losses_list = []
                baseline_losses_list = []
                valid_k = []

                for k in k_values:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    metrics = task_metrics.get(combo_key)
                    if metrics is None:
                        continue
                    if p not in metrics.maml_loss_at_steps:
                        continue

                    maml_losses_list.append(metrics.maml_loss_at_steps[p])
                    baseline_losses_list.append(metrics.baseline_loss_at_steps[p])
                    valid_k.append(k)

                if len(valid_k) >= 2:
                    fig = plot_sample_efficiency(
                        k_values=np.array(valid_k),
                        maml_losses=np.array(maml_losses_list),
                        baseline_losses=np.array(baseline_losses_list),
                        fixed_step=p,
                        title=f"{task_name}: Sample Efficiency (noise={noise:.0%}, step {p})",
                        save_path=task_dir
                        / f"sample_efficiency_noise{noise:.2f}_step{p}.png",
                        dpi=dpi,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 7: Jacobian histogram (one per group per K × noise)
        # Groups with multiple members overlay estimates (e.g. NS nu_u + nu_v)
        # ---------------------------------------------------------------------
        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"
                combo_data = task_samples.get(combo_key, {})

                maml_pe = combo_data.get("maml_pred_errors")
                baseline_pe = combo_data.get("baseline_pred_errors")

                for coeff_name, members in coeff_group.items():
                    member_names = [m["name"] for m in members]
                    output_indices = [m["output_index"] for m in members]

                    maml_ests = [combo_data.get(f"maml_{n}") for n in member_names]
                    baseline_ests = [
                        combo_data.get(f"baseline_{n}") for n in member_names
                    ]
                    true_arr = combo_data.get(f"maml_{member_names[0]}_true")

                    if true_arr is None or any(
                        x is None for x in maml_ests + baseline_ests
                    ):
                        continue
                    if any(len(x) == 0 for x in maml_ests + baseline_ests):
                        continue

                    maml_pe_list = [
                        maml_pe[:, idx] if maml_pe is not None else None
                        for idx in output_indices
                    ]
                    baseline_pe_list = [
                        baseline_pe[:, idx] if baseline_pe is not None else None
                        for idx in output_indices
                    ]

                    fig = plot_jacobian_histogram(
                        maml_estimates=maml_ests,
                        baseline_estimates=baseline_ests,
                        estimate_labels=member_names,
                        coeff_true=float(true_arr[0]),
                        title=f"{task_name}: {coeff_name} Coefficient (K={k}, noise={noise:.0%})",
                        coeff_name=coeff_name,
                        save_path=task_dir
                        / f"jacobian_histogram_{coeff_name}_k{k}_noise{noise:.2f}{_combo_worse_suffix(task_data, combo_key)}.png",
                        dpi=dpi,
                        maml_pred_errors=maml_pe_list,
                        baseline_pred_errors=baseline_pe_list,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 8: Coefficient error heatmap (one per group)
        # ---------------------------------------------------------------------
        for coeff_name, members in coeff_group.items():
            member_names = [m["name"] for m in members]
            true_key = f"{member_names[0]}_true"
            maml_errors = np.full((len(noise_levels), len(k_values)), np.nan)
            baseline_errors = np.full((len(noise_levels), len(k_values)), np.nan)
            has_jacobian_data = False

            for i, noise in enumerate(noise_levels):
                for j, k in enumerate(k_values):
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo_data = task_samples.get(combo_key, {})

                    maml_ests = [combo_data.get(f"maml_{n}") for n in member_names]
                    baseline_ests = [
                        combo_data.get(f"baseline_{n}") for n in member_names
                    ]
                    coeff_true = combo_data.get(f"maml_{true_key}")

                    if coeff_true is None or any(x is None for x in maml_ests):
                        continue
                    has_jacobian_data = True
                    true_val = float(coeff_true[0])
                    if true_val == 0:
                        continue

                    maml_recovered = float(np.mean([np.mean(x) for x in maml_ests]))
                    maml_errors[i, j] = abs(maml_recovered - true_val) / true_val * 100

                    if not any(x is None for x in baseline_ests):
                        baseline_recovered = float(
                            np.mean([np.mean(x) for x in baseline_ests])
                        )
                        baseline_errors[i, j] = (
                            abs(baseline_recovered - true_val) / true_val * 100
                        )

            if has_jacobian_data:
                fig = plot_coefficient_heatmap(
                    k_values=k_values,
                    noise_levels=noise_levels,
                    maml_errors=maml_errors,
                    baseline_errors=baseline_errors,
                    title=f"{task_name}: {coeff_name} Recovery Error (%)",
                    save_path=task_dir / f"coefficient_heatmap_{coeff_name}.png",
                    dpi=dpi,
                )
                plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 9: Coefficient recovery vs K (one per noise level, per group)
        # ---------------------------------------------------------------------
        for coeff_name, members in coeff_group.items():
            member_names = [m["name"] for m in members]
            true_key = f"{member_names[0]}_true"

            for noise in noise_levels:
                maml_err_list: list[float] = []
                baseline_err_list: list[float] = []
                valid_k: list[int] = []

                for k in k_values:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo_data = task_samples.get(combo_key, {})

                    maml_ests = [combo_data.get(f"maml_{n}") for n in member_names]
                    baseline_ests = [
                        combo_data.get(f"baseline_{n}") for n in member_names
                    ]
                    coeff_true = combo_data.get(f"maml_{true_key}")

                    if coeff_true is None or any(
                        x is None for x in maml_ests + baseline_ests
                    ):
                        continue
                    true_val = float(coeff_true[0])
                    if true_val == 0:
                        continue

                    maml_recovered = float(np.mean([np.mean(x) for x in maml_ests]))
                    baseline_recovered = float(
                        np.mean([np.mean(x) for x in baseline_ests])
                    )
                    maml_err_list.append(
                        abs(maml_recovered - true_val) / true_val * 100
                    )
                    baseline_err_list.append(
                        abs(baseline_recovered - true_val) / true_val * 100
                    )
                    valid_k.append(k)

                if len(valid_k) >= 2:
                    fig = plot_coefficient_vs_k(
                        k_values=np.array(valid_k),
                        maml_errors=np.array(maml_err_list),
                        baseline_errors=np.array(baseline_err_list),
                        title=f"{task_name}: {coeff_name} Error vs K (noise={noise:.0%})",
                        save_path=task_dir
                        / f"coefficient_vs_k_{coeff_name}_noise{noise:.2f}.png",
                        dpi=dpi,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 10: Coefficient recovery vs noise (one per K, per group)
        # ---------------------------------------------------------------------
        for coeff_name, members in coeff_group.items():
            member_names = [m["name"] for m in members]
            true_key = f"{member_names[0]}_true"

            for k in k_values:
                maml_err_list = []
                baseline_err_list = []
                valid_noise: list[float] = []

                for noise in noise_levels:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo_data = task_samples.get(combo_key, {})

                    maml_ests = [combo_data.get(f"maml_{n}") for n in member_names]
                    baseline_ests = [
                        combo_data.get(f"baseline_{n}") for n in member_names
                    ]
                    coeff_true = combo_data.get(f"maml_{true_key}")

                    if coeff_true is None or any(
                        x is None for x in maml_ests + baseline_ests
                    ):
                        continue
                    true_val = float(coeff_true[0])
                    if true_val == 0:
                        continue

                    maml_recovered = float(np.mean([np.mean(x) for x in maml_ests]))
                    baseline_recovered = float(
                        np.mean([np.mean(x) for x in baseline_ests])
                    )
                    maml_err_list.append(
                        abs(maml_recovered - true_val) / true_val * 100
                    )
                    baseline_err_list.append(
                        abs(baseline_recovered - true_val) / true_val * 100
                    )
                    valid_noise.append(noise)

                if len(valid_noise) >= 2:
                    fig = plot_coefficient_vs_noise(
                        noise_levels=np.array(valid_noise),
                        maml_errors=np.array(maml_err_list),
                        baseline_errors=np.array(baseline_err_list),
                        title=f"{task_name}: {coeff_name} Error vs Noise (K={k})",
                        save_path=task_dir
                        / f"coefficient_vs_noise_{coeff_name}_k{k}.png",
                        dpi=dpi,
                    )
                    plt.close(fig)


def generate_aggregated_figures(
    results: dict,
    all_metrics: Dict[str, Dict[str, ComparisonMetrics]],
    k_values: NDArray[np.integer[Any]],
    fixed_steps: NDArray[np.integer[Any]],
    noise_levels: NDArray[np.floating[Any]],
    output_dir: Path,
    dpi: int,
    deriv_threshold: float = 1e-7,
    holdout_size: int = 1000,
) -> None:
    """Generate aggregated figures (mean ± std across tasks)."""

    # Build coefficient grouping from any task's specs (structure is PDE-constant)
    any_task = next(iter(results["tasks"].values()), {})
    specs = any_task.get("coefficient_specs", [])
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in specs:
        grouped.setdefault(s["coeff_name"], []).append(s)

    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    task_names = list(all_metrics.keys())
    n_tasks = len(task_names)

    if n_tasks == 0:
        print("  No valid results to aggregate.")
        return

    print(f"  Aggregating across {n_tasks} tasks...")

    # -------------------------------------------------------------------------
    # Aggregate speedup heatmap
    # -------------------------------------------------------------------------
    speedup_stack = []
    maml_loss_stack = []
    baseline_loss_stack = []

    for task_name in task_names:
        task_metrics = all_metrics[task_name]
        speedups = np.full((len(noise_levels), len(k_values)), np.nan)
        maml_losses = np.full((len(noise_levels), len(k_values)), np.nan)
        baseline_losses = np.full((len(noise_levels), len(k_values)), np.nan)

        for i, noise in enumerate(noise_levels):
            for j, k in enumerate(k_values):
                combo_key = f"k_{k}_noise_{noise:.2f}"
                metrics = task_metrics.get(combo_key)
                if metrics is not None:
                    speedups[i, j] = metrics.speedup
                    maml_losses[i, j] = metrics.maml_plateau_loss
                    baseline_losses[i, j] = metrics.baseline_plateau_loss

        speedup_stack.append(speedups)
        maml_loss_stack.append(maml_losses)
        baseline_loss_stack.append(baseline_losses)

    speedup_stack = np.array(speedup_stack)

    # Count infs and compute stats on finite values only
    inf_counts = np.sum(np.isinf(speedup_stack), axis=0)
    speedup_stack_clean = np.where(np.isinf(speedup_stack), np.nan, speedup_stack)
    speedup_mean = np.nanmean(speedup_stack_clean, axis=0)
    speedup_std = np.nanstd(speedup_stack_clean, axis=0)

    # For aggregated, show mean of losses
    # maml_loss_mean = np.nanmean(np.array(maml_loss_stack), axis=0)
    # baseline_loss_mean = np.nanmean(np.array(baseline_loss_stack), axis=0)

    fig = plot_speedup_heatmap(
        speedups=speedup_mean,
        k_values=k_values,
        noise_levels=noise_levels,
        title=f"Aggregated Speedup Ratio (n={n_tasks} tasks)",
        save_path=agg_dir / "speedup_heatmap.png",
        dpi=dpi,
        std_values=speedup_std,
        inf_counts=inf_counts,
        n_total=n_tasks,
    )
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate loss ratio heatmaps
    # -------------------------------------------------------------------------
    for p in fixed_steps:
        ratio_stack = []
        for task_name in task_names:
            task_metrics = all_metrics[task_name]
            ratios = np.full((len(noise_levels), len(k_values)), np.nan)

            for i, noise in enumerate(noise_levels):
                for j, k in enumerate(k_values):
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    metrics = task_metrics.get(combo_key)
                    if metrics is not None and p in metrics.loss_ratios:
                        ratios[i, j] = metrics.loss_ratios[p]

            ratio_stack.append(ratios)

        ratio_stack = np.array(ratio_stack)

        # Count infs and compute stats on finite values only
        ratio_inf_counts = np.sum(np.isinf(ratio_stack), axis=0)
        ratio_stack_clean = np.where(np.isinf(ratio_stack), np.nan, ratio_stack)
        ratio_mean = np.nanmean(ratio_stack_clean, axis=0)
        ratio_std: NDArray[np.floating[Any]] = np.nanstd(ratio_stack_clean, axis=0)

        fig = plot_loss_ratio_heatmap(
            ratios=ratio_mean,
            k_values=k_values,
            noise_levels=noise_levels,
            fixed_step=p,
            title=f"Aggregated Loss Ratio @ step {p} (n={n_tasks} tasks)",
            save_path=agg_dir / f"loss_ratio_heatmap_step{p}.png",
            dpi=dpi,
            std_values=ratio_std,
            inf_counts=ratio_inf_counts,
            n_total=n_tasks,
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate convergence plots (one per K × noise)
    # -------------------------------------------------------------------------
    for k in k_values:
        for noise in noise_levels:
            combo_key = f"k_{k}_noise_{noise:.2f}"

            # Collect curves and min steps from all tasks
            maml_curves = []
            baseline_curves = []
            maml_min_steps = []
            baseline_min_steps = []

            for task_name in task_names:
                task_data = results["tasks"][task_name]["samples"].get(combo_key, {})
                maml_losses = task_data.get("maml_losses")
                baseline_losses = task_data.get("baseline_losses")

                if maml_losses is not None and baseline_losses is not None:
                    maml_curves.append(maml_losses)
                    baseline_curves.append(baseline_losses)
                    maml_min_steps.append(steps_to_plateau(maml_losses))
                    baseline_min_steps.append(steps_to_plateau(baseline_losses))

            if len(maml_curves) < 2:
                continue

            # Align curve lengths
            min_len = min(
                min(len(c) for c in maml_curves), min(len(c) for c in baseline_curves)
            )
            maml_curves = [c[:min_len] for c in maml_curves]
            baseline_curves = [c[:min_len] for c in baseline_curves]

            maml_arr = np.array(maml_curves)
            baseline_arr = np.array(baseline_curves)

            # Compute mean and std of min steps
            maml_mean_step = int(np.mean(maml_min_steps))
            baseline_mean_step = int(np.mean(baseline_min_steps))
            maml_step_std = float(np.std(maml_min_steps))
            baseline_step_std = float(np.std(baseline_min_steps))

            fig = plot_convergence(
                maml_losses=np.mean(maml_arr, axis=0),
                baseline_losses=np.mean(baseline_arr, axis=0),
                title=f"Aggregated: K={k}, noise={noise:.0%} (n={len(maml_curves)} tasks)",
                save_path=agg_dir / f"convergence_k{k}_noise{noise:.2f}.png",
                dpi=dpi,
                maml_std=np.std(maml_arr, axis=0),
                baseline_std=np.std(baseline_arr, axis=0),
                maml_min_step=maml_mean_step,
                baseline_min_step=baseline_mean_step,
                min_step_std=(maml_step_std, baseline_step_std),
                deriv_threshold=deriv_threshold,
                k_shot=k,
            )
            plt.close(fig)

            # Aggregated train-holdout convergence (if holdout data available)
            maml_train_curves = []
            maml_holdout_curves = []
            baseline_train_curves = []
            baseline_holdout_curves = []

            for task_name in task_names:
                task_data = results["tasks"][task_name]["samples"].get(combo_key, {})
                mt = task_data.get("maml_losses")
                mh = task_data.get("maml_holdout_losses")
                bt = task_data.get("baseline_losses")
                bh = task_data.get("baseline_holdout_losses")

                if (
                    mt is not None
                    and mh is not None
                    and bt is not None
                    and bh is not None
                ):
                    maml_train_curves.append(mt)
                    maml_holdout_curves.append(mh)
                    baseline_train_curves.append(bt)
                    baseline_holdout_curves.append(bh)

            if len(maml_train_curves) > 0:
                # Truncate to min length
                min_len = min(
                    min(len(c) for c in maml_train_curves),
                    min(len(c) for c in maml_holdout_curves),
                    min(len(c) for c in baseline_train_curves),
                    min(len(c) for c in baseline_holdout_curves),
                )
                maml_train_curves = [c[:min_len] for c in maml_train_curves]
                maml_holdout_curves = [c[:min_len] for c in maml_holdout_curves]
                baseline_train_curves = [c[:min_len] for c in baseline_train_curves]
                baseline_holdout_curves = [c[:min_len] for c in baseline_holdout_curves]

                mt_arr = np.array(maml_train_curves)
                mh_arr = np.array(maml_holdout_curves)
                bt_arr = np.array(baseline_train_curves)
                bh_arr = np.array(baseline_holdout_curves)

                fig = plot_train_holdout_convergence(
                    maml_train=np.mean(mt_arr, axis=0),
                    maml_holdout=np.mean(mh_arr, axis=0),
                    baseline_train=np.mean(bt_arr, axis=0),
                    baseline_holdout=np.mean(bh_arr, axis=0),
                    title=f"Aggregated Train vs Holdout: K={k}, noise={noise:.0%} (n={len(maml_train_curves)} tasks)",
                    save_path=agg_dir / f"train_holdout_k{k}_noise{noise:.2f}.png",
                    dpi=dpi,
                    k_shot=k,
                    holdout_size=holdout_size,
                    maml_train_std=np.std(mt_arr, axis=0),
                    maml_holdout_std=np.std(mh_arr, axis=0),
                    baseline_train_std=np.std(bt_arr, axis=0),
                    baseline_holdout_std=np.std(bh_arr, axis=0),
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate noise robustness curves
    # -------------------------------------------------------------------------
    for k in k_values:
        maml_steps_per_task = []
        baseline_steps_per_task = []

        for task_name in task_names:
            task_metrics = all_metrics[task_name]
            maml_row = []
            baseline_row = []

            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"
                metrics = task_metrics.get(combo_key)

                if metrics is None:
                    maml_row.append(np.nan)
                    baseline_row.append(np.nan)
                else:
                    # Use steps_to_plateau (primary metric)
                    maml_row.append(metrics.maml_steps_to_plateau)
                    baseline_row.append(metrics.baseline_steps_to_plateau)

            maml_steps_per_task.append(maml_row)
            baseline_steps_per_task.append(baseline_row)

        maml_arr = np.array(maml_steps_per_task)
        baseline_arr = np.array(baseline_steps_per_task)

        fig = plot_noise_robustness(
            noise_levels=noise_levels,
            maml_steps=np.nanmean(maml_arr, axis=0),
            baseline_steps=np.nanmean(baseline_arr, axis=0),
            title=f"Aggregated Noise Robustness (K={k}, n={n_tasks} tasks)",
            save_path=agg_dir / f"noise_robustness_k{k}.png",
            dpi=dpi,
            maml_std=np.nanstd(maml_arr, axis=0),
            baseline_std=np.nanstd(baseline_arr, axis=0),
        )
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate sample efficiency curves
    # -------------------------------------------------------------------------
    for p in fixed_steps[:1]:
        for noise in noise_levels:
            maml_per_task = []
            baseline_per_task = []

            for task_name in task_names:
                task_metrics = all_metrics[task_name]
                maml_row = []
                baseline_row = []

                for k in k_values:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    metrics = task_metrics.get(combo_key)

                    if metrics is None or p not in metrics.maml_loss_at_steps:
                        maml_row.append(np.nan)
                        baseline_row.append(np.nan)
                    else:
                        maml_row.append(metrics.maml_loss_at_steps[p])
                        baseline_row.append(metrics.baseline_loss_at_steps[p])

                maml_per_task.append(maml_row)
                baseline_per_task.append(baseline_row)

            maml_arr = np.array(maml_per_task)
            baseline_arr = np.array(baseline_per_task)

            fig = plot_sample_efficiency(
                k_values=k_values,
                maml_losses=np.nanmean(maml_arr, axis=0),
                baseline_losses=np.nanmean(baseline_arr, axis=0),
                fixed_step=p,
                title=f"Aggregated Sample Efficiency (noise={noise:.0%}, step {p}, n={n_tasks} tasks)",
                save_path=agg_dir / f"sample_efficiency_noise{noise:.2f}_step{p}.png",
                dpi=dpi,
                maml_std=np.nanstd(maml_arr, axis=0),
                baseline_std=np.nanstd(baseline_arr, axis=0),
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate Jacobian histograms (Graph 7) - one per group per K × noise
    # Ratio-normalized: each task's estimates divided by that task's true value
    # -------------------------------------------------------------------------
    for k in k_values:
        for noise in noise_levels:
            combo_key = f"k_{k}_noise_{noise:.2f}"

            for coeff_name, members in grouped.items():
                member_names = [m["name"] for m in members]
                maml_all: list[list[float]] = [[] for _ in members]
                baseline_all: list[list[float]] = [[] for _ in members]
                has_data = False

                for task_name in task_names:
                    task_data = results["tasks"][task_name]["samples"].get(
                        combo_key, {}
                    )
                    true_arr = task_data.get(f"maml_{member_names[0]}_true")
                    if true_arr is None:
                        continue
                    true_val = float(true_arr[0])
                    if true_val == 0:
                        continue

                    for i, name in enumerate(member_names):
                        m = task_data.get(f"maml_{name}")
                        b = task_data.get(f"baseline_{name}")
                        if m is not None:
                            has_data = True
                            maml_all[i].extend(m.flatten() / true_val)
                        if b is not None:
                            baseline_all[i].extend(b.flatten() / true_val)

                if has_data and all(len(x) > 0 for x in maml_all + baseline_all):
                    fig = plot_jacobian_histogram(
                        maml_estimates=[np.array(x) for x in maml_all],
                        baseline_estimates=[np.array(x) for x in baseline_all],
                        estimate_labels=member_names,
                        coeff_true=1.0,
                        title=f"Aggregated {coeff_name} Recovery Ratio (K={k}, noise={noise:.0%}, n={n_tasks} tasks)",
                        coeff_name=coeff_name,
                        save_path=agg_dir
                        / f"jacobian_histogram_{coeff_name}_k{k}_noise{noise:.2f}.png",
                        dpi=dpi,
                        ratio_mode=True,
                    )
                    plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate coefficient error heatmap (Graph 8) - one per group
    # -------------------------------------------------------------------------
    for coeff_name, members in grouped.items():
        member_names = [m["name"] for m in members]
        true_key = f"{member_names[0]}_true"
        maml_error_stack = []
        baseline_error_stack = []
        has_jacobian_data = False

        for task_name in task_names:
            task_samples = results["tasks"][task_name]["samples"]
            maml_errors = np.full((len(noise_levels), len(k_values)), np.nan)
            baseline_errors = np.full((len(noise_levels), len(k_values)), np.nan)

            for i, noise in enumerate(noise_levels):
                for j, k in enumerate(k_values):
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo_data = task_samples.get(combo_key, {})

                    maml_ests = [combo_data.get(f"maml_{n}") for n in member_names]
                    baseline_ests = [
                        combo_data.get(f"baseline_{n}") for n in member_names
                    ]
                    coeff_true = combo_data.get(f"maml_{true_key}")

                    if coeff_true is None or any(x is None for x in maml_ests):
                        continue
                    has_jacobian_data = True
                    true_val = float(coeff_true[0])
                    if true_val == 0:
                        continue

                    maml_recovered = float(np.mean([np.mean(x) for x in maml_ests]))
                    maml_errors[i, j] = abs(maml_recovered - true_val) / true_val * 100

                    if not any(x is None for x in baseline_ests):
                        baseline_recovered = float(
                            np.mean([np.mean(x) for x in baseline_ests])
                        )
                        baseline_errors[i, j] = (
                            abs(baseline_recovered - true_val) / true_val * 100
                        )

            maml_error_stack.append(maml_errors)
            baseline_error_stack.append(baseline_errors)

        if has_jacobian_data:
            maml_error_mean = np.nanmean(np.array(maml_error_stack), axis=0)
            baseline_error_mean = np.nanmean(np.array(baseline_error_stack), axis=0)

            fig = plot_coefficient_heatmap(
                k_values=k_values,
                noise_levels=noise_levels,
                maml_errors=maml_error_mean,
                baseline_errors=baseline_error_mean,
                title=f"Aggregated {coeff_name} Recovery Error (%, n={n_tasks} tasks)",
                save_path=agg_dir / f"coefficient_heatmap_{coeff_name}.png",
                dpi=dpi,
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate coefficient vs K (Graph 9) - one per noise level × group
    # -------------------------------------------------------------------------
    for coeff_name, members in grouped.items():
        member_names = [m["name"] for m in members]
        true_key = f"{member_names[0]}_true"

        for noise in noise_levels:
            maml_per_task = []
            baseline_per_task = []

            for task_name in task_names:
                task_samples = results["tasks"][task_name]["samples"]
                maml_row: list[float] = []
                baseline_row: list[float] = []

                for k in k_values:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo_data = task_samples.get(combo_key, {})

                    maml_ests = [combo_data.get(f"maml_{n}") for n in member_names]
                    baseline_ests = [
                        combo_data.get(f"baseline_{n}") for n in member_names
                    ]
                    coeff_true = combo_data.get(f"maml_{true_key}")

                    if coeff_true is None or any(
                        x is None for x in maml_ests + baseline_ests
                    ):
                        maml_row.append(np.nan)
                        baseline_row.append(np.nan)
                    else:
                        true_val = float(coeff_true[0])
                        if true_val == 0:
                            maml_row.append(np.nan)
                            baseline_row.append(np.nan)
                        else:
                            maml_recovered = float(
                                np.mean([np.mean(x) for x in maml_ests])
                            )
                            baseline_recovered = float(
                                np.mean([np.mean(x) for x in baseline_ests])
                            )
                            maml_row.append(
                                abs(maml_recovered - true_val) / true_val * 100
                            )
                            baseline_row.append(
                                abs(baseline_recovered - true_val) / true_val * 100
                            )

                maml_per_task.append(maml_row)
                baseline_per_task.append(baseline_row)

            maml_arr = np.array(maml_per_task)
            baseline_arr = np.array(baseline_per_task)

            if not np.all(np.isnan(maml_arr)):
                fig = plot_coefficient_vs_k(
                    k_values=k_values,
                    maml_errors=np.nanmean(maml_arr, axis=0),
                    baseline_errors=np.nanmean(baseline_arr, axis=0),
                    title=f"Aggregated {coeff_name} Error vs K (noise={noise:.0%}, n={n_tasks} tasks)",
                    save_path=agg_dir
                    / f"coefficient_vs_k_{coeff_name}_noise{noise:.2f}.png",
                    dpi=dpi,
                    maml_std=np.nanstd(maml_arr, axis=0),
                    baseline_std=np.nanstd(baseline_arr, axis=0),
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate coefficient vs noise (Graph 10) - one per K × group
    # -------------------------------------------------------------------------
    for coeff_name, members in grouped.items():
        member_names = [m["name"] for m in members]
        true_key = f"{member_names[0]}_true"

        for k in k_values:
            maml_per_task = []
            baseline_per_task = []

            for task_name in task_names:
                task_samples = results["tasks"][task_name]["samples"]
                maml_row = []
                baseline_row = []

                for noise in noise_levels:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo_data = task_samples.get(combo_key, {})

                    maml_ests = [combo_data.get(f"maml_{n}") for n in member_names]
                    baseline_ests = [
                        combo_data.get(f"baseline_{n}") for n in member_names
                    ]
                    coeff_true = combo_data.get(f"maml_{true_key}")

                    if coeff_true is None or any(
                        x is None for x in maml_ests + baseline_ests
                    ):
                        maml_row.append(np.nan)
                        baseline_row.append(np.nan)
                    else:
                        true_val = float(coeff_true[0])
                        if true_val == 0:
                            maml_row.append(np.nan)
                            baseline_row.append(np.nan)
                        else:
                            maml_recovered = float(
                                np.mean([np.mean(x) for x in maml_ests])
                            )
                            baseline_recovered = float(
                                np.mean([np.mean(x) for x in baseline_ests])
                            )
                            maml_row.append(
                                abs(maml_recovered - true_val) / true_val * 100
                            )
                            baseline_row.append(
                                abs(baseline_recovered - true_val) / true_val * 100
                            )

                maml_per_task.append(maml_row)
                baseline_per_task.append(baseline_row)

            maml_arr = np.array(maml_per_task)
            baseline_arr = np.array(baseline_per_task)

            if not np.all(np.isnan(maml_arr)):
                fig = plot_coefficient_vs_noise(
                    noise_levels=noise_levels,
                    maml_errors=np.nanmean(maml_arr, axis=0),
                    baseline_errors=np.nanmean(baseline_arr, axis=0),
                    title=f"Aggregated {coeff_name} Error vs Noise (K={k}, n={n_tasks} tasks)",
                    save_path=agg_dir / f"coefficient_vs_noise_{coeff_name}_k{k}.png",
                    dpi=dpi,
                    maml_std=np.nanstd(maml_arr, axis=0),
                    baseline_std=np.nanstd(baseline_arr, axis=0),
                )
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from MAML evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to results.json (default: auto-detect from experiment dir)",
    )
    parser.add_argument(
        "--no-per-task", action="store_true", help="Skip per-task figure generation"
    )
    parser.add_argument(
        "--no-aggregated", action="store_true", help="Skip aggregated figure generation"
    )
    args = parser.parse_args()

    # =========================================================================
    # Load configuration
    # =========================================================================
    print("=" * 60)
    print("MAML Visualization")
    print("=" * 60)
    print()

    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    exp_dir = (
        Path("data/models") / config.get("output", {}).get("base_dir", "") / exp_name
    )

    print(f"Experiment: {exp_name}")
    print()

    # =========================================================================
    # Start processing
    # =========================================================================
    eval_cfg = config.get("evaluation", {})
    viz_cfg = config.get("visualization", {})
    dpi = viz_cfg.get("dpi", 150)

    eval_base_dir = exp_dir / "evaluation"
    results_path = eval_base_dir / "results.json"
    output_dir = exp_dir / "figures"

    if not results_path.exists():
        raise ValueError("Results doesn't exist!")

    print("=" * 60)
    print(f"Processing {results_path}")
    print("=" * 60)
    print()

    # Load results
    print("-" * 60)
    print("Loading results...")
    print("-" * 60)

    results = load_results_with_samples(results_path)
    print(f"Loaded from: {results_path}")
    print(f"Tasks: {len(results['tasks'])}")
    print()

    # Get parameters
    k_values: NDArray[np.integer[Any]] = np.array(
        results["config"].get(
            "k_values", eval_cfg.get("k_values", np.array([10, 50, 100, 500, 1000]))
        )
    )
    noise_levels: NDArray[np.floating[Any]] = np.array(
        results["config"].get(
            "noise_levels",
            eval_cfg.get("noise_levels", np.array([0.0, 0.01, 0.05, 0.10])),
        )
    )
    fixed_steps: NDArray[np.integer[Any]] = np.array(
        results["config"].get(
            "fixed_steps", eval_cfg.get("fixed_steps", np.array([50, 100, 200]))
        )
    )
    deriv_threshold = float(eval_cfg.get("deriv_threshold", 1e-7))
    holdout_size = int(
        results["config"].get("holdout_size", eval_cfg.get("holdout_size", 1000))
    )
    # Compute metrics
    print("-" * 60)
    print("Computing metrics...")
    print("-" * 60)

    all_metrics = compute_all_metrics(results, fixed_steps, deriv_threshold)
    print(f"Computed metrics for {len(all_metrics)} tasks")
    print()

    # Generate figures
    if not args.no_per_task:
        print("-" * 60)
        print("Generating per-task figures...")
        print("-" * 60)
        generate_per_task_figures(
            results=results,
            all_metrics=all_metrics,
            k_values=k_values,
            noise_levels=noise_levels,
            fixed_steps=fixed_steps,
            output_dir=output_dir,
            dpi=dpi,
            deriv_threshold=deriv_threshold,
            holdout_size=holdout_size,
        )
        print()

    if not args.no_aggregated:
        print("-" * 60)
        print("Generating aggregated figures...")
        print("-" * 60)
        generate_aggregated_figures(
            results=results,
            all_metrics=all_metrics,
            k_values=k_values,
            noise_levels=noise_levels,
            fixed_steps=fixed_steps,
            output_dir=output_dir,
            dpi=dpi,
            deriv_threshold=deriv_threshold,
            holdout_size=holdout_size,
        )
        print()

    print(f"Figures saved to: {output_dir}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print()
    print("Output directory:")
    print("\t" + str(output_dir))
    print()


if __name__ == "__main__":
    main()
