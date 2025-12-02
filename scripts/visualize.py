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

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_comparison_metrics, ComparisonMetrics, steps_to_plateau
from src.evaluation.graphs import (
    plot_convergence,
    plot_train_holdout_convergence,
    plot_speedup_heatmap,
    plot_loss_ratio_heatmap,
    plot_noise_robustness,
    plot_sample_efficiency,
)


def load_config(config_path: Path) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results(results_path: Path) -> dict:
    """Load evaluation results (metadata only)."""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_curves(curves_dir: Path, task_name: str) -> Dict[str, np.ndarray]:
    """
    Load loss curves from NPZ file for a task.

    Args:
        curves_dir: Directory containing .npz curve files
        task_name: Name of the task

    Returns:
        Dict mapping combo_key/curve_name to array, or empty dict if not found
    """
    npz_path = curves_dir / f"{task_name}.npz"
    if not npz_path.exists():
        return {}
    return dict(np.load(npz_path))


def load_results_with_curves(results_path: Path) -> dict:
    """
    Load evaluation results with curves from NPZ files.

    Supports both old format (curves in JSON) and new format (curves in NPZ).

    Args:
        results_path: Path to results.json

    Returns:
        Results dict with curves populated in task_data['curves']
    """
    results = load_results(results_path)

    # Check for curves directory (new format)
    curves_dir = results_path.parent / 'curves'

    for task_name, task_data in results['tasks'].items():
        # New format: load from NPZ
        if curves_dir.exists():
            task_curves = load_curves(curves_dir, task_name)
            if task_curves:
                # Restructure curves into old format for compatibility
                task_data['results'] = {}
                # Group by combo_key
                combo_keys = set()
                for key in task_curves.keys():
                    # Keys are like "k_100_noise_0.01/maml_train_losses"
                    combo_key = key.rsplit('/', 1)[0]
                    combo_keys.add(combo_key)

                for combo_key in combo_keys:
                    task_data['results'][combo_key] = {
                        'maml_losses': task_curves.get(f"{combo_key}/maml_train_losses"),
                        'baseline_losses': task_curves.get(f"{combo_key}/baseline_train_losses"),
                        'maml_holdout_losses': task_curves.get(f"{combo_key}/maml_holdout_losses"),
                        'baseline_holdout_losses': task_curves.get(f"{combo_key}/baseline_holdout_losses"),
                    }
                    # Convert to lists for compatibility with existing code
                    for k, v in task_data['results'][combo_key].items():
                        if v is not None:
                            task_data['results'][combo_key][k] = v.tolist()

        # Old format: curves already in 'results' dict (backwards compatibility)
        # Nothing to do - already in correct format

    return results


def compute_all_metrics(
    results: dict,
    fixed_steps: List[int],
    deriv_threshold: float = 1e-7,
) -> Dict[str, Dict[str, ComparisonMetrics]]:
    """
    Compute metrics for all (task, K, noise) combinations.

    Args:
        results: Loaded results data (with curves)
        threshold: L* threshold for legacy convergence detection
        fixed_steps: Steps at which to record loss values
        deriv_threshold: Maximum |derivative| for plateau detection

    Returns:
        Dict mapping task_name -> combo_key -> ComparisonMetrics
    """
    all_metrics = {}

    for task_name, task_data in results['tasks'].items():
        all_metrics[task_name] = {}

        if 'results' not in task_data:
            continue

        for combo_key, combo_data in task_data['results'].items():
            maml_losses = combo_data.get('maml_losses')
            baseline_losses = combo_data.get('baseline_losses')

            if maml_losses is None or baseline_losses is None:
                continue

            # Get holdout losses if available
            maml_holdout = combo_data.get('maml_holdout_losses')
            baseline_holdout = combo_data.get('baseline_holdout_losses')

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
    k_values: List[int],
    noise_levels: List[float],
    fixed_steps: List[int],
    output_dir: Path,
    dpi: int,
    deriv_threshold: float = 1e-7,
    holdout_size: int = 1000,
) -> None:
    """Generate all per-task figures."""

    for task_name, task_data in results['tasks'].items():
        task_dir = output_dir / 'per_task' / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        task_metrics = all_metrics.get(task_name, {})
        task_results = task_data['results']

        print(f"  Generating figures for {task_name}...")

        # ---------------------------------------------------------------------
        # Graph 5: Convergence plots (one per K × noise)
        # ---------------------------------------------------------------------
        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"
                combo_data = task_results.get(combo_key, {})

                maml_losses = combo_data.get('maml_losses')
                baseline_losses = combo_data.get('baseline_losses')

                if maml_losses is None or baseline_losses is None:
                    continue

                fig = plot_convergence(
                    maml_losses=maml_losses,
                    baseline_losses=baseline_losses,
                    title=f"{task_name}: K={k}, noise={noise:.0%}",
                    save_path=task_dir / f"convergence_k{k}_noise{noise:.2f}.png",
                    dpi=dpi,
                    deriv_threshold=deriv_threshold,
                    k_shot=k,
                )
                plt.close(fig)

                # Train vs Holdout comparison (if holdout data available)
                maml_holdout = combo_data.get('maml_holdout_losses')
                baseline_holdout = combo_data.get('baseline_holdout_losses')
                if maml_holdout is not None and baseline_holdout is not None:
                    fig = plot_train_holdout_convergence(
                        maml_train=maml_losses,
                        maml_holdout=maml_holdout,
                        baseline_train=baseline_losses,
                        baseline_holdout=baseline_holdout,
                        title=f"{task_name}: K={k}, noise={noise:.0%} (Train vs Holdout)",
                        save_path=task_dir / f"train_holdout_k{k}_noise{noise:.2f}.png",
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
                    noise_levels=valid_noise,
                    maml_steps=maml_steps_list,
                    baseline_steps=baseline_steps_list,
                    k_value=k,
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
                        k_values=valid_k,
                        maml_losses=maml_losses_list,
                        baseline_losses=baseline_losses_list,
                        noise_level=noise,
                        fixed_step=p,
                        title=f"{task_name}: Sample Efficiency (noise={noise:.0%}, step {p})",
                        save_path=task_dir / f"sample_efficiency_noise{noise:.2f}_step{p}.png",
                        dpi=dpi,
                    )
                    plt.close(fig)


def generate_aggregated_figures(
    results: dict,
    all_metrics: Dict[str, Dict[str, ComparisonMetrics]],
    k_values: List[int],
    noise_levels: List[float],
    fixed_steps: List[int],
    output_dir: Path,
    dpi: int,
    deriv_threshold: float = 1e-7,
    holdout_size: int = 1000,
) -> None:
    """Generate aggregated figures (mean ± std across tasks)."""

    agg_dir = output_dir / 'aggregated'
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
    maml_loss_mean = np.nanmean(np.array(maml_loss_stack), axis=0)
    baseline_loss_mean = np.nanmean(np.array(baseline_loss_stack), axis=0)

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
        ratio_std = np.nanstd(ratio_stack_clean, axis=0)

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
                task_data = results['tasks'][task_name]['results'].get(combo_key, {})
                maml_losses = task_data.get('maml_losses')
                baseline_losses = task_data.get('baseline_losses')

                if maml_losses is not None and baseline_losses is not None:
                    maml_curves.append(maml_losses)
                    baseline_curves.append(baseline_losses)
                    maml_min_steps.append(steps_to_plateau(maml_losses))
                    baseline_min_steps.append(steps_to_plateau(baseline_losses))

            if len(maml_curves) < 2:
                continue

            # Align curve lengths
            min_len = min(min(len(c) for c in maml_curves), min(len(c) for c in baseline_curves))
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
                maml_losses=np.mean(maml_arr, axis=0).tolist(),
                baseline_losses=np.mean(baseline_arr, axis=0).tolist(),
                title=f"Aggregated: K={k}, noise={noise:.0%} (n={len(maml_curves)} tasks)",
                save_path=agg_dir / f"convergence_k{k}_noise{noise:.2f}.png",
                dpi=dpi,
                maml_std=np.std(maml_arr, axis=0).tolist(),
                baseline_std=np.std(baseline_arr, axis=0).tolist(),
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
                task_data = results['tasks'][task_name]['results'].get(combo_key, {})
                mt = task_data.get('maml_losses')
                mh = task_data.get('maml_holdout_losses')
                bt = task_data.get('baseline_losses')
                bh = task_data.get('baseline_holdout_losses')

                if mt is not None and mh is not None and bt is not None and bh is not None:
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
                    maml_train=np.mean(mt_arr, axis=0).tolist(),
                    maml_holdout=np.mean(mh_arr, axis=0).tolist(),
                    baseline_train=np.mean(bt_arr, axis=0).tolist(),
                    baseline_holdout=np.mean(bh_arr, axis=0).tolist(),
                    title=f"Aggregated Train vs Holdout: K={k}, noise={noise:.0%} (n={len(maml_train_curves)} tasks)",
                    save_path=agg_dir / f"train_holdout_k{k}_noise{noise:.2f}.png",
                    dpi=dpi,
                    k_shot=k,
                    holdout_size=holdout_size,
                    maml_train_std=np.std(mt_arr, axis=0).tolist(),
                    maml_holdout_std=np.std(mh_arr, axis=0).tolist(),
                    baseline_train_std=np.std(bt_arr, axis=0).tolist(),
                    baseline_holdout_std=np.std(bh_arr, axis=0).tolist(),
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
            maml_steps=np.nanmean(maml_arr, axis=0).tolist(),
            baseline_steps=np.nanmean(baseline_arr, axis=0).tolist(),
            k_value=k,
            title=f"Aggregated Noise Robustness (K={k}, n={n_tasks} tasks)",
            save_path=agg_dir / f"noise_robustness_k{k}.png",
            dpi=dpi,
            maml_std=np.nanstd(maml_arr, axis=0).tolist(),
            baseline_std=np.nanstd(baseline_arr, axis=0).tolist(),
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
                maml_losses=np.nanmean(maml_arr, axis=0).tolist(),
                baseline_losses=np.nanmean(baseline_arr, axis=0).tolist(),
                noise_level=noise,
                fixed_step=p,
                title=f"Aggregated Sample Efficiency (noise={noise:.0%}, step {p}, n={n_tasks} tasks)",
                save_path=agg_dir / f"sample_efficiency_noise{noise:.2f}_step{p}.png",
                dpi=dpi,
                maml_std=np.nanstd(maml_arr, axis=0).tolist(),
                baseline_std=np.nanstd(baseline_arr, axis=0).tolist(),
            )
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations from MAML evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', type=Path, required=True,
        help='Path to experiment YAML config'
    )
    parser.add_argument(
        '--results', type=Path, default=None,
        help='Path to results.json (default: auto-detect from experiment dir)'
    )
    parser.add_argument(
        '--no-per-task', action='store_true',
        help='Skip per-task figure generation'
    )
    parser.add_argument(
        '--no-aggregated', action='store_true',
        help='Skip aggregated figure generation'
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
    exp_name = config['experiment']['name']
    exp_dir = Path('data/models') / config.get('output', {}).get('base_dir', '') / exp_name

    print(f"Experiment: {exp_name}")
    print()

    # =========================================================================
    # Discover target modes (noisy_targets, clean_targets)
    # =========================================================================
    eval_base_dir = exp_dir / 'evaluation'

    if args.results is not None:
        # User specified a specific results file
        target_modes = [("custom", args.results, exp_dir / 'figures')]
    else:
        # Auto-detect target mode directories
        target_modes = []

        # Check for new structure (separate subdirs)
        for mode_name in ["noisy_targets", "clean_targets"]:
            mode_dir = eval_base_dir / mode_name
            results_path = mode_dir / 'results.json'
            if results_path.exists():
                target_modes.append((mode_name, results_path, exp_dir / 'figures' / mode_name))

        # Fallback to old structure (single results.json)
        if not target_modes:
            old_results_path = eval_base_dir / 'results.json'
            if old_results_path.exists():
                target_modes.append(("default", old_results_path, exp_dir / 'figures'))

        if not target_modes:
            raise FileNotFoundError(
                f"No results found in: {eval_base_dir}\n"
                f"Run evaluate.py first."
            )

    print(f"Found {len(target_modes)} evaluation mode(s)")
    print()

    # =========================================================================
    # Process each target mode
    # =========================================================================
    eval_cfg = config.get('evaluation', {})
    viz_cfg = config.get('visualization', {})
    dpi = viz_cfg.get('dpi', 150)

    for mode_name, results_path, output_dir in target_modes:
        print("=" * 60)
        print(f"Processing: {mode_name}")
        print("=" * 60)
        print()

        # Load results
        print("-" * 60)
        print("Loading results...")
        print("-" * 60)

        results = load_results_with_curves(results_path)
        print(f"Loaded from: {results_path}")
        print(f"Tasks: {len(results['tasks'])}")
        print()

        # Get parameters
        k_values = results['config'].get('k_values', eval_cfg.get('k_values', [10, 50, 100, 500, 1000]))
        noise_levels = results['config'].get('noise_levels', eval_cfg.get('noise_levels', [0.0, 0.01, 0.05, 0.10]))
        fixed_steps = results['config'].get('fixed_steps', eval_cfg.get('fixed_steps', [50, 100, 200]))
        deriv_threshold = eval_cfg.get('deriv_threshold', 1e-7)
        holdout_size = results['config'].get('holdout_size', eval_cfg.get('holdout_size', 1000))

        # Format mode label for titles
        if mode_name == "noisy_targets":
            mode_label = " [Noisy Targets]"
        elif mode_name == "clean_targets":
            mode_label = " [Clean Targets]"
        else:
            mode_label = ""

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
    print(f"Output directories:")
    for mode_name, _, output_dir in target_modes:
        print(f"  {mode_name}: {output_dir}")
    print()


if __name__ == '__main__':
    main()
