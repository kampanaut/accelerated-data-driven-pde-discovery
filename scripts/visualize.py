#!/usr/bin/env python3
"""
Generate visualizations from MAML evaluation results.

Usage:
    python scripts/visualize.py --config configs/experiment.yaml
    python scripts/visualize.py --config configs/experiment.yaml --only scatter,best-combo
    python scripts/visualize.py --config configs/experiment.yaml --only jacobian[1,1000]
    python scripts/visualize.py --config configs/experiment.yaml --only scatter[500](br-18,heat-1)
"""

import dataclasses
import json
import re
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple

from numpy.typing import NDArray
import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    compute_comparison_metrics,
    ComparisonMetrics,
    compress_step_ranges,
    steps_to_plateau,
)
VALID_GRAPH_NAMES = {
    "jacobian",
    "generalization",
    "loss-ratio",
    "noise-robustness",
    "sample-efficiency",
    "coeff-heatmap",
    "coeff-vs-k",
    "coeff-vs-noise",
    "scatter",
    "best-combo",
}

METRICS_GRAPHS = {"generalization", "loss-ratio", "noise-robustness", "sample-efficiency"}

STEP_FILTERABLE = {"jacobian", "scatter", "loss-ratio"}

EXPERIMENT_FILTERABLE = {"scatter"}


@dataclasses.dataclass
class GraphSelection:
    """Controls which graphs to generate and optional per-step/experiment filtering."""

    graphs: set[str]
    step_filters: dict[str, list[int]]
    experiment_filters: dict[str, list[str]]
    all_graphs: bool  # True when --only not provided

    def enabled(self, name: str) -> bool:
        return self.all_graphs or (name in self.graphs)

    def steps_for(self, name: str, all_steps: list[int]) -> list[int]:
        if name not in self.step_filters:
            return all_steps
        requested = self.step_filters[name]
        available = set(all_steps)
        missing = [s for s in requested if s not in available]
        if missing:
            print(f"  Skipping {name} steps {missing}: not in available steps {all_steps}")
        return [s for s in requested if s in available]

    def experiments_for(self, name: str) -> list[str] | None:
        """Return explicit experiment list, or None for default discovery."""
        return self.experiment_filters.get(name)

    def needs_metrics(self) -> bool:
        if self.all_graphs:
            return True
        return bool(self.graphs & METRICS_GRAPHS)


def parse_only(arg: str | None) -> GraphSelection:
    """
    Parse --only argument into a GraphSelection.

    Syntax: comma-separated graph names with optional [step,...] and/or
    (experiment,...) suffixes.

    Examples:
        "jacobian[1,1000],scatter[500],best-combo"
        "scatter[1,25,1000](br-18,heat-1,br-19)"
        "scatter(br-18,heat-1)"

    Returns GraphSelection with all_graphs=True when arg is None.
    """
    if arg is None:
        return GraphSelection(
            graphs=set(), step_filters={}, experiment_filters={}, all_graphs=True
        )

    graphs: set[str] = set()
    step_filters: dict[str, list[int]] = {}
    experiment_filters: dict[str, list[str]] = {}
    pattern = re.compile(r"^([a-z-]+)(?:\[([0-9,]+)\])?(?:\(([^)]+)\))?$")

    # Split on commas NOT inside brackets or parentheses
    tokens = re.split(r",(?![^[]*\])(?![^(]*\))", arg)

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        m = pattern.match(token)
        if m is None:
            raise ValueError(
                f"Invalid --only token: {token!r}. "
                f"Expected format: name, name[step,...], or name(exp,...)\n"
                f"Valid names: {', '.join(sorted(VALID_GRAPH_NAMES))}"
            )

        name = m.group(1)
        if name not in VALID_GRAPH_NAMES:
            raise ValueError(
                f"Unknown graph name: {name!r}. "
                f"Valid names: {', '.join(sorted(VALID_GRAPH_NAMES))}"
            )

        graphs.add(name)

        if m.group(2) is not None:
            if name not in STEP_FILTERABLE:
                raise ValueError(
                    f"Step filtering not supported for {name!r}. "
                    f"Only {', '.join(sorted(STEP_FILTERABLE))} accept [steps]."
                )
            steps = [int(s) for s in m.group(2).split(",")]
            existing = step_filters.get(name, [])
            step_filters[name] = sorted(set(existing + steps))

        if m.group(3) is not None:
            if name not in EXPERIMENT_FILTERABLE:
                raise ValueError(
                    f"Experiment filtering not supported for {name!r}. "
                    f"Only {', '.join(sorted(EXPERIMENT_FILTERABLE))} accept (experiments)."
                )
            exps = [e.strip() for e in m.group(3).split(",")]
            existing_exps = experiment_filters.get(name, [])
            experiment_filters[name] = list(dict.fromkeys(existing_exps + exps))

    return GraphSelection(
        graphs=graphs,
        step_filters=step_filters,
        experiment_filters=experiment_filters,
        all_graphs=False,
    )


from src.evaluation.graphs import (
    plot_train_holdout_convergence,
    plot_loss_ratio_heatmap,
    plot_noise_robustness,
    plot_sample_efficiency,
    # Graph 7-10: Jacobian analysis graphs
    plot_jacobian_histogram,
    plot_coefficient_heatmap,
    plot_coefficient_vs_k,
    plot_coefficient_vs_noise,
    # Graph 11: Cross-experiment scatter
    plot_coefficient_scatter_grid,
    PanelDataDict,
    # Graph 12: Best-combo prediction scatter
    plot_best_combo_scatter,
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


def _loss_worse_suffix(loss_worse_steps: list[int], all_fixed_steps: list[int]) -> str:
    """Build WORSE(LOSS[...]) suffix for loss-group figures."""
    if not loss_worse_steps:
        return ""
    compressed = compress_step_ranges(loss_worse_steps, all_fixed_steps)
    return f"_WORSE(LOSS[{compressed}])"


def _coeff_worse_suffix_at_step(
    coeff_worse_steps: dict[str, list[int]], step: int
) -> str:
    """Build WORSE(COEFF:name) suffix for a specific step (Jacobian histograms)."""
    flags = []
    for name, steps in coeff_worse_steps.items():
        if step in steps:
            flags.append(f"COEFF:{name}")
    if flags:
        return f"_WORSE({','.join(flags)})"
    return ""


def _task_dir_worse_suffix(task_data: dict, all_fixed_steps: list[int]) -> str:
    """Build task-level directory WORSE suffix with step ranges."""
    flags = []

    loss_steps = task_data.get("loss_worse_steps", [])
    if loss_steps:
        compressed = compress_step_ranges(loss_steps, all_fixed_steps)
        flags.append(f"LOSS[{compressed}]")

    coeff_steps = task_data.get("coeff_worse_steps", {})
    for name, steps in coeff_steps.items():
        if steps:
            compressed = compress_step_ranges(steps, all_fixed_steps)
            flags.append(f"COEFF:{name}[{compressed}]")

    if flags:
        return f"_WORSE({','.join(flags)})"
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

        # Discover combo_keys from NPZ key prefixes (skip best_combo prefix)
        combo_keys = set()
        for key in raw.keys():
            prefix = key.rsplit("/")[0]
            if prefix != "best_combo":
                combo_keys.add(prefix)

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

        # Extract best_combo data (stored with best_combo/ prefix in NPZ)
        bc_preds = raw.get("best_combo/predictions")
        if bc_preds is not None:
            for bc_key in raw:
                if bc_key.startswith("best_combo/"):
                    flat_key = bc_key.replace("/", "_")
                    task_data["samples"][flat_key] = raw[bc_key]

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
    sel: GraphSelection = GraphSelection(
        graphs=set(), step_filters={}, experiment_filters={}, all_graphs=True
    ),
) -> None:
    """Generate all per-task figures."""

    for task_name, task_data in results["tasks"].items():
        suffix = _task_dir_worse_suffix(task_data, fixed_steps.tolist())

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
        # Graph 7: Jacobian histogram (one per group per K × noise × step)
        # Groups with multiple members overlay estimates (e.g. NS nu_u + nu_v)
        # ---------------------------------------------------------------------
        if sel.enabled("jacobian"):
            for k in k_values:
                for noise in noise_levels:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo_data = task_samples.get(combo_key, {})
                    combo_worse = task_data.get(f"worse_{combo_key}", {})
                    combo_coeff_worse = combo_worse.get("coeff_steps", {})

                    combo_fixed_steps = (
                        fixed_steps  # from function param (config-level)
                    )
                    maml_pe = combo_data.get("maml_pred_errors")
                    baseline_pe = combo_data.get("baseline_pred_errors")

                    for coeff_name, members in coeff_group.items():
                        member_names = [m["name"] for m in members]
                        output_indices = [m["output_index"] for m in members]

                        maml_ests_full = [
                            combo_data.get(f"maml_{n}") for n in member_names
                        ]
                        baseline_ests_full = [
                            combo_data.get(f"baseline_{n}") for n in member_names
                        ]
                        true_arr = combo_data.get(f"maml_{member_names[0]}_true")

                        if true_arr is None or any(
                            x is None for x in maml_ests_full + baseline_ests_full
                        ):
                            continue

                        # Determine step indices to iterate
                        if (
                            combo_fixed_steps is not None
                            and maml_ests_full[0].ndim == 2
                        ):
                            allowed = sel.steps_for(
                                "jacobian", combo_fixed_steps.tolist()
                            )
                            step_indices = [
                                i
                                for i, s in enumerate(combo_fixed_steps)
                                if int(s) in allowed
                            ]
                        else:
                            # Legacy: single-step data
                            step_indices = [0]
                            combo_fixed_steps = None

                        for si in step_indices:
                            step_label = (
                                int(combo_fixed_steps[si])
                                if combo_fixed_steps is not None
                                else ""
                            )

                            # Index into per-step arrays
                            if combo_fixed_steps is not None:
                                maml_ests = [x[si] for x in maml_ests_full]
                                baseline_ests = [x[si] for x in baseline_ests_full]
                                maml_pe_step = (
                                    maml_pe[si] if maml_pe is not None else None
                                )
                                baseline_pe_step = (
                                    baseline_pe[si] if baseline_pe is not None else None
                                )
                            else:
                                maml_ests = maml_ests_full
                                baseline_ests = baseline_ests_full
                                maml_pe_step = maml_pe
                                baseline_pe_step = baseline_pe

                            if any(len(x) == 0 for x in maml_ests + baseline_ests):
                                continue

                            maml_pe_list = [
                                maml_pe_step[:, idx]
                                if maml_pe_step is not None
                                else None
                                for idx in output_indices
                            ]
                            baseline_pe_list = [
                                baseline_pe_step[:, idx]
                                if baseline_pe_step is not None
                                else None
                                for idx in output_indices
                            ]

                            step_suffix = (
                                f"_step[{step_label}]" if step_label != "" else ""
                            )
                            coeff_worse = (
                                _coeff_worse_suffix_at_step(
                                    combo_coeff_worse, int(step_label)
                                )
                                if step_label != ""
                                else ""
                            )
                            fig = plot_jacobian_histogram(
                                maml_estimates=maml_ests,
                                baseline_estimates=baseline_ests,
                                estimate_labels=member_names,
                                coeff_true=float(true_arr[0]),
                                title=f"{task_name}: {coeff_name} (K={k}, noise={noise:.0%}, step {step_label})",
                                coeff_name=coeff_name,
                                save_path=task_dir
                                / f"coeff[{coeff_name}]_k[{k}]_noise[{noise:.2f}]{step_suffix}_distribution{coeff_worse}.png",
                                dpi=dpi,
                                maml_pred_errors=maml_pe_list,
                                baseline_pred_errors=baseline_pe_list,
                            )
                            plt.close(fig)

        # ---------------------------------------------------------------------
        # Generalization plot (one per K × noise)
        # ---------------------------------------------------------------------
        if sel.enabled("generalization"):
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

                    maml_holdout = np.array(combo_data.get("maml_holdout_losses"))
                    baseline_holdout = np.array(
                        combo_data.get("baseline_holdout_losses")
                    )
                    if maml_holdout is None or baseline_holdout is None:
                        continue

                    # Build LOSS-only worse suffix with step list
                    combo_worse = task_data.get(f"worse_{combo_key}", {})
                    combo_loss_steps = combo_worse.get("loss_steps", [])
                    loss_suffix = _loss_worse_suffix(
                        combo_loss_steps, fixed_steps.tolist()
                    )

                    fs_list = fixed_steps.tolist()

                    fig = plot_train_holdout_convergence(
                        maml_train=maml_losses,
                        maml_holdout=maml_holdout,
                        baseline_train=baseline_losses,
                        baseline_holdout=baseline_holdout,
                        title=f"{task_name}: K={k}, noise={noise:.0%}",
                        save_path=task_dir
                        / f"k[{k}]_noise[{noise:.2f}]_finetune_generalization{loss_suffix}.png",
                        dpi=dpi,
                        k_shot=k,
                        holdout_size=holdout_size,
                        deriv_threshold=deriv_threshold,
                        fixed_steps=fs_list,
                        loss_worse_steps=combo_loss_steps,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 4: Loss ratio heatmap (per fixed step)
        # ---------------------------------------------------------------------
        if sel.enabled("loss-ratio"):
            for p in sel.steps_for("loss-ratio", fixed_steps.tolist()):
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
                    save_path=task_dir / f"step[{p}]_loss_ratio_heatmap.png",
                    dpi=dpi,
                )
                plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 1: Noise robustness curves (one per K)
        # ---------------------------------------------------------------------
        if sel.enabled("noise-robustness"):
            for k in k_values:
                maml_steps_list = []
                baseline_steps_list = []
                valid_noise = []

                for noise in noise_levels:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    metrics = task_metrics.get(combo_key)
                    if metrics is None:
                        continue

                    maml_steps_list.append(metrics.maml_steps_to_plateau)
                    baseline_steps_list.append(metrics.baseline_steps_to_plateau)
                    valid_noise.append(noise)

                if len(valid_noise) >= 2:
                    fig = plot_noise_robustness(
                        noise_levels=np.array(valid_noise),
                        maml_steps=np.array(maml_steps_list),
                        baseline_steps=np.array(baseline_steps_list),
                        title=f"{task_name}: Noise Robustness (K={k})",
                        save_path=task_dir / f"k[{k}]_noise_robustness.png",
                        dpi=dpi,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 2: Sample efficiency curves (one per noise level)
        # ---------------------------------------------------------------------
        if sel.enabled("sample-efficiency"):
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
                            / f"noise[{noise:.2f}]_step[{p}]_sample_efficiency.png",
                            dpi=dpi,
                        )
                        plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 8: Coefficient error heatmap (one per group, final step)
        # ---------------------------------------------------------------------
        if sel.enabled("coeff-heatmap"):
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

                        maml_ests_raw = [
                            combo_data.get(f"maml_{n}") for n in member_names
                        ]
                        baseline_ests_raw = [
                            combo_data.get(f"baseline_{n}") for n in member_names
                        ]
                        coeff_true = combo_data.get(f"maml_{true_key}")

                        if coeff_true is None or any(
                            x is None for x in maml_ests_raw
                        ):
                            continue
                        has_jacobian_data = True
                        true_val = float(coeff_true[0])
                        if true_val == 0:
                            continue

                        maml_ests = [
                            x[-1] if x.ndim == 2 else x for x in maml_ests_raw
                        ]
                        maml_recovered = float(
                            np.mean([np.mean(x) for x in maml_ests])
                        )
                        maml_errors[i, j] = (
                            abs(maml_recovered - true_val) / true_val * 100
                        )

                        if not any(x is None for x in baseline_ests_raw):
                            baseline_ests = [
                                x[-1] if x.ndim == 2 else x for x in baseline_ests_raw
                            ]
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
                        save_path=task_dir
                        / f"coeff[{coeff_name}]_error_heatmap.png",
                        dpi=dpi,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 9: Coefficient recovery vs K
        # ---------------------------------------------------------------------
        if sel.enabled("coeff-vs-k"):
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

                        maml_ests_raw = [
                            combo_data.get(f"maml_{n}") for n in member_names
                        ]
                        baseline_ests_raw = [
                            combo_data.get(f"baseline_{n}") for n in member_names
                        ]
                        coeff_true = combo_data.get(f"maml_{true_key}")

                        if coeff_true is None or any(
                            x is None for x in maml_ests_raw + baseline_ests_raw
                        ):
                            continue
                        true_val = float(coeff_true[0])
                        if true_val == 0:
                            continue

                        maml_ests = [
                            x[-1] if x.ndim == 2 else x for x in maml_ests_raw
                        ]
                        baseline_ests = [
                            x[-1] if x.ndim == 2 else x for x in baseline_ests_raw
                        ]
                        maml_recovered = float(
                            np.mean([np.mean(x) for x in maml_ests])
                        )
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
                            / f"coeff[{coeff_name}]_noise[{noise:.2f}]_error_vs_k.png",
                            dpi=dpi,
                        )
                        plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 10: Coefficient recovery vs noise
        # ---------------------------------------------------------------------
        if sel.enabled("coeff-vs-noise"):
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

                        maml_ests_raw = [
                            combo_data.get(f"maml_{n}") for n in member_names
                        ]
                        baseline_ests_raw = [
                            combo_data.get(f"baseline_{n}") for n in member_names
                        ]
                        coeff_true = combo_data.get(f"maml_{true_key}")

                        if coeff_true is None or any(
                            x is None for x in maml_ests_raw + baseline_ests_raw
                        ):
                            continue
                        true_val = float(coeff_true[0])
                        if true_val == 0:
                            continue

                        maml_ests = [
                            x[-1] if x.ndim == 2 else x for x in maml_ests_raw
                        ]
                        baseline_ests = [
                            x[-1] if x.ndim == 2 else x for x in baseline_ests_raw
                        ]
                        maml_recovered = float(
                            np.mean([np.mean(x) for x in maml_ests])
                        )
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
                            / f"coeff[{coeff_name}]_k[{k}]_error_vs_noise.png",
                            dpi=dpi,
                        )
                        plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 12: Best-combo prediction scatter
        # ---------------------------------------------------------------------
        if not sel.enabled("best-combo"):
            continue
        bc_predictions = task_samples.get("best_combo_predictions")
        if bc_predictions is not None:
            bc_true = task_samples.get("best_combo_true_targets")
            bc_x = task_samples.get("best_combo_x_pts")
            bc_y = task_samples.get("best_combo_y_pts")
            bc_steps = task_samples.get("best_combo_steps")
            bc_errors = task_samples.get("best_combo_coeff_error")
            bc_key_arr = task_samples.get("best_combo_key")

            if all(v is not None for v in [bc_true, bc_x, bc_y, bc_steps, bc_errors, bc_key_arr]):
                assert bc_true is not None and bc_x is not None and bc_y is not None
                assert bc_steps is not None and bc_errors is not None and bc_key_arr is not None

                # Parse combo key for filename tags
                bc_key_str = str(bc_key_arr)
                # bc_key_str is like "k_800_noise_0.00"
                k_match = re.search(r"k_(\d+)", bc_key_str)
                n_match = re.search(r"noise_([\d.]+)", bc_key_str)
                k_tag = k_match.group(1) if k_match else "?"
                n_tag = n_match.group(1) if n_match else "?"

                n_outputs = bc_predictions.shape[2]
                output_labels = ["u_t"] if n_outputs == 1 else ["u_t", "v_t"]

                for oi, olabel in enumerate(output_labels):
                    suffix = f"_{olabel}" if n_outputs > 1 else ""
                    save_name = (
                        f"k[{k_tag}]_noise[{n_tag}]_best_combo_prediction_scatter{suffix}.png"
                    )

                    fig = plot_best_combo_scatter(
                        predictions=bc_predictions,
                        true_targets=bc_true,
                        x_pts=bc_x,
                        y_pts=bc_y,
                        steps=bc_steps,
                        coeff_errors=bc_errors,
                        output_index=oi,
                        output_label=olabel,
                        title=f"{task_name}: Prediction Evolution ({bc_key_str}, {olabel})",
                        save_path=task_dir / save_name,
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
    sel: GraphSelection = GraphSelection(
        graphs=set(), step_filters={}, experiment_filters={}, all_graphs=True
    ),
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
    # Aggregate Jacobian histograms (Graph 7) - one per group per K × noise × step
    # Ratio-normalized: each task's estimates divided by that task's true value
    # -------------------------------------------------------------------------
    if sel.enabled("jacobian"):
        # Determine step indices from first available combo
        agg_fixed_steps = None
        for task_name in task_names:
            any_combo = next(
                iter(results["tasks"][task_name].get("samples", {}).values()), {}
            )
            fs = any_combo.get("fixed_steps")
            if fs is not None:
                agg_fixed_steps = fs
                break

        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"

                # Determine per-step iteration with filtering
                if agg_fixed_steps is not None:
                    allowed = sel.steps_for(
                        "jacobian", [int(s) for s in agg_fixed_steps]
                    )
                    step_indices = [
                        i
                        for i, s in enumerate(agg_fixed_steps)
                        if int(s) in allowed
                    ]
                else:
                    step_indices = [0]

                for si in step_indices:
                    step_label = (
                        int(agg_fixed_steps[si]) if agg_fixed_steps is not None else ""
                    )

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
                                    m_step = m[si] if m.ndim == 2 else m
                                    maml_all[i].extend(m_step.flatten() / true_val)
                                if b is not None:
                                    b_step = b[si] if b.ndim == 2 else b
                                    baseline_all[i].extend(b_step.flatten() / true_val)

                        if has_data and all(
                            len(x) > 0 for x in maml_all + baseline_all
                        ):
                            step_suffix = (
                                f"_step[{step_label}]" if step_label != "" else ""
                            )
                            fig = plot_jacobian_histogram(
                                maml_estimates=[np.array(x) for x in maml_all],
                                baseline_estimates=[np.array(x) for x in baseline_all],
                                estimate_labels=member_names,
                                coeff_true=1.0,
                                title=f"Aggregated {coeff_name} Recovery Ratio (K={k}, noise={noise:.0%}, step {step_label}, n={n_tasks} tasks)",
                                coeff_name=coeff_name,
                                save_path=agg_dir
                                / f"coeff[{coeff_name}]_k[{k}]_noise[{noise:.2f}]{step_suffix}_distribution.png",
                                dpi=dpi,
                                ratio_mode=True,
                            )
                            plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate loss ratio heatmaps
    # -------------------------------------------------------------------------
    if sel.enabled("loss-ratio"):
        for p in sel.steps_for("loss-ratio", fixed_steps.tolist()):
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

            ratio_stack_arr = np.array(ratio_stack)

            ratio_inf_counts = np.sum(np.isinf(ratio_stack_arr), axis=0)
            ratio_stack_clean = np.where(
                np.isinf(ratio_stack_arr), np.nan, ratio_stack_arr
            )
            ratio_mean = np.nanmean(ratio_stack_clean, axis=0)
            ratio_std: NDArray[np.floating[Any]] = np.nanstd(
                ratio_stack_clean, axis=0
            )

            fig = plot_loss_ratio_heatmap(
                ratios=ratio_mean,
                k_values=k_values,
                noise_levels=noise_levels,
                fixed_step=p,
                title=f"Aggregated Loss Ratio @ step {p} (n={n_tasks} tasks)",
                save_path=agg_dir / f"step[{p}]_loss_ratio_heatmap.png",
                dpi=dpi,
                std_values=ratio_std,
                inf_counts=ratio_inf_counts,
                n_total=n_tasks,
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate generalization plots (one per K × noise)
    # -------------------------------------------------------------------------
    if sel.enabled("generalization"):
        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"

                maml_train_curves = []
                maml_holdout_curves = []
                baseline_train_curves = []
                baseline_holdout_curves = []
                maml_plateau_steps = []
                baseline_plateau_steps = []

                for task_name in task_names:
                    task_data = results["tasks"][task_name]["samples"].get(
                        combo_key, {}
                    )
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
                        maml_plateau_steps.append(steps_to_plateau(mh))
                        baseline_plateau_steps.append(steps_to_plateau(bh))

                if len(maml_train_curves) == 0:
                    continue

                min_len = min(
                    min(len(c) for c in maml_train_curves),
                    min(len(c) for c in maml_holdout_curves),
                    min(len(c) for c in baseline_train_curves),
                    min(len(c) for c in baseline_holdout_curves),
                )
                maml_train_curves = [c[:min_len] for c in maml_train_curves]
                maml_holdout_curves = [c[:min_len] for c in maml_holdout_curves]
                baseline_train_curves = [c[:min_len] for c in baseline_train_curves]
                baseline_holdout_curves = [
                    c[:min_len] for c in baseline_holdout_curves
                ]

                mt_arr = np.array(maml_train_curves)
                mh_arr = np.array(maml_holdout_curves)
                bt_arr = np.array(baseline_train_curves)
                bh_arr = np.array(baseline_holdout_curves)

                fig = plot_train_holdout_convergence(
                    maml_train=np.mean(mt_arr, axis=0),
                    maml_holdout=np.mean(mh_arr, axis=0),
                    baseline_train=np.mean(bt_arr, axis=0),
                    baseline_holdout=np.mean(bh_arr, axis=0),
                    title=f"Aggregated: K={k}, noise={noise:.0%} (n={len(maml_train_curves)} tasks)",
                    save_path=agg_dir
                    / f"k[{k}]_noise[{noise:.2f}]_finetune_generalization.png",
                    dpi=dpi,
                    k_shot=k,
                    holdout_size=holdout_size,
                    maml_train_std=np.std(mt_arr, axis=0),
                    maml_holdout_std=np.std(mh_arr, axis=0),
                    baseline_train_std=np.std(bt_arr, axis=0),
                    baseline_holdout_std=np.std(bh_arr, axis=0),
                    deriv_threshold=deriv_threshold,
                    maml_plateau_step=int(np.mean(maml_plateau_steps)),
                    baseline_plateau_step=int(np.mean(baseline_plateau_steps)),
                    plateau_step_std=(
                        float(np.std(maml_plateau_steps)),
                        float(np.std(baseline_plateau_steps)),
                    ),
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate noise robustness curves
    # -------------------------------------------------------------------------
    if sel.enabled("noise-robustness"):
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
                save_path=agg_dir / f"k[{k}]_noise_robustness.png",
                dpi=dpi,
                maml_std=np.nanstd(maml_arr, axis=0),
                baseline_std=np.nanstd(baseline_arr, axis=0),
            )
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate sample efficiency curves
    # -------------------------------------------------------------------------
    if sel.enabled("sample-efficiency"):
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
                    save_path=agg_dir
                    / f"noise[{noise:.2f}]_step[{p}]_sample_efficiency.png",
                    dpi=dpi,
                    maml_std=np.nanstd(maml_arr, axis=0),
                    baseline_std=np.nanstd(baseline_arr, axis=0),
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate coefficient error heatmap (Graph 8)
    # -------------------------------------------------------------------------
    if sel.enabled("coeff-heatmap"):
        for coeff_name, members in grouped.items():
            member_names = [m["name"] for m in members]
            true_key = f"{member_names[0]}_true"
            maml_error_stack = []
            baseline_error_stack = []
            has_jacobian_data = False

            for task_name in task_names:
                task_samples = results["tasks"][task_name]["samples"]
                maml_errors = np.full((len(noise_levels), len(k_values)), np.nan)
                baseline_errors = np.full(
                    (len(noise_levels), len(k_values)), np.nan
                )

                for i, noise in enumerate(noise_levels):
                    for j, k in enumerate(k_values):
                        combo_key = f"k_{k}_noise_{noise:.2f}"
                        combo_data = task_samples.get(combo_key, {})

                        maml_ests_raw = [
                            combo_data.get(f"maml_{n}") for n in member_names
                        ]
                        baseline_ests_raw = [
                            combo_data.get(f"baseline_{n}") for n in member_names
                        ]
                        coeff_true = combo_data.get(f"maml_{true_key}")

                        if coeff_true is None or any(
                            x is None for x in maml_ests_raw
                        ):
                            continue
                        has_jacobian_data = True
                        true_val = float(coeff_true[0])
                        if true_val == 0:
                            continue

                        maml_ests = [
                            x[-1] if x.ndim == 2 else x for x in maml_ests_raw
                        ]
                        maml_recovered = float(
                            np.mean([np.mean(x) for x in maml_ests])
                        )
                        maml_errors[i, j] = (
                            abs(maml_recovered - true_val) / true_val * 100
                        )

                        if not any(x is None for x in baseline_ests_raw):
                            baseline_ests = [
                                x[-1] if x.ndim == 2 else x
                                for x in baseline_ests_raw
                            ]
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
                baseline_error_mean = np.nanmean(
                    np.array(baseline_error_stack), axis=0
                )

                fig = plot_coefficient_heatmap(
                    k_values=k_values,
                    noise_levels=noise_levels,
                    maml_errors=maml_error_mean,
                    baseline_errors=baseline_error_mean,
                    title=f"Aggregated {coeff_name} Recovery Error (%, n={n_tasks} tasks)",
                    save_path=agg_dir / f"coeff[{coeff_name}]_error_heatmap.png",
                    dpi=dpi,
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate coefficient vs K (Graph 9)
    # -------------------------------------------------------------------------
    if sel.enabled("coeff-vs-k"):
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

                        maml_ests_raw = [
                            combo_data.get(f"maml_{n}") for n in member_names
                        ]
                        baseline_ests_raw = [
                            combo_data.get(f"baseline_{n}") for n in member_names
                        ]
                        coeff_true = combo_data.get(f"maml_{true_key}")

                        if coeff_true is None or any(
                            x is None for x in maml_ests_raw + baseline_ests_raw
                        ):
                            maml_row.append(np.nan)
                            baseline_row.append(np.nan)
                        else:
                            true_val = float(coeff_true[0])
                            if true_val == 0:
                                maml_row.append(np.nan)
                                baseline_row.append(np.nan)
                            else:
                                maml_ests = [
                                    x[-1] if x.ndim == 2 else x
                                    for x in maml_ests_raw
                                ]
                                baseline_ests = [
                                    x[-1] if x.ndim == 2 else x
                                    for x in baseline_ests_raw
                                ]
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
                                    abs(baseline_recovered - true_val)
                                    / true_val
                                    * 100
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
                        / f"coeff[{coeff_name}]_noise[{noise:.2f}]_error_vs_k.png",
                        dpi=dpi,
                        maml_std=np.nanstd(maml_arr, axis=0),
                        baseline_std=np.nanstd(baseline_arr, axis=0),
                    )
                    plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate coefficient vs noise (Graph 10)
    # -------------------------------------------------------------------------
    if sel.enabled("coeff-vs-noise"):
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

                        maml_ests_raw = [
                            combo_data.get(f"maml_{n}") for n in member_names
                        ]
                        baseline_ests_raw = [
                            combo_data.get(f"baseline_{n}") for n in member_names
                        ]
                        coeff_true = combo_data.get(f"maml_{true_key}")

                        if coeff_true is None or any(
                            x is None for x in maml_ests_raw + baseline_ests_raw
                        ):
                            maml_row.append(np.nan)
                            baseline_row.append(np.nan)
                        else:
                            true_val = float(coeff_true[0])
                            if true_val == 0:
                                maml_row.append(np.nan)
                                baseline_row.append(np.nan)
                            else:
                                maml_ests = [
                                    x[-1] if x.ndim == 2 else x
                                    for x in maml_ests_raw
                                ]
                                baseline_ests = [
                                    x[-1] if x.ndim == 2 else x
                                    for x in baseline_ests_raw
                                ]
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
                                    abs(baseline_recovered - true_val)
                                    / true_val
                                    * 100
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
                        save_path=agg_dir
                        / f"coeff[{coeff_name}]_k[{k}]_error_vs_noise.png",
                        dpi=dpi,
                        maml_std=np.nanstd(maml_arr, axis=0),
                        baseline_std=np.nanstd(baseline_arr, axis=0),
                    )
                    plt.close(fig)


# ─── Cross-experiment coefficient scatter ─────────────────────────────────────


def resolve_experiment_names(
    config: dict, experiment_names: list[str]
) -> list[Tuple[str, Path]]:
    """
    Resolve CLI-provided experiment names to (name, results_path) pairs.

    Args:
        config: Experiment config (for base_dir)
        experiment_names: List of experiment directory names from --only scatter(...)

    Returns:
        List of (dir_name, results_json_path) sorted by dir_name
    """
    models_root = Path(config.get("output", {}).get("base_dir", "data/models"))

    experiments: list[Tuple[str, Path]] = []
    missing: list[str] = []

    for name in experiment_names:
        results_path = models_root / name / "evaluation" / "results.json"
        if results_path.exists():
            experiments.append((name, results_path))
        else:
            missing.append(name)

    if missing:
        raise ValueError(
            f"Experiments not found (no evaluation/results.json): {', '.join(missing)}"
        )

    result = sorted(experiments, key=lambda x: x[0])
    print(f"  CLI-selected {len(result)} experiments for scatter:")
    for name, _ in result:
        print(f"    - {name}")

    return result


def discover_comparison_experiments(
    config: dict, current_exp_dir: Path
) -> list[Tuple[str, Path]]:
    """
    Discover experiments to compare in the coefficient scatter plot.

    If visualization.compare_experiments is set, use those dir names.
    Otherwise auto-discover all data/models/* with matching pde_type
    and an existing evaluation/results.json.

    Returns list of (dir_name, results_json_path) sorted by dir_name.
    """
    current_pde_type = config["experiment"]["pde_type"]
    models_root = Path(config.get("output", {}).get("base_dir", "data/models"))

    viz_cfg = config.get("visualization", {})
    explicit = viz_cfg.get("compare_experiments", [])

    # Suffix exclusion: regexes tested against dir name. Capture group = iteration count.
    exclude_patterns = [r"-ISNAN$", r"-ENDNAN@(\d+)$"] + viz_cfg.get(
        "exclude_suffixes_append", []
    )
    exclude_re = [re.compile(p) for p in exclude_patterns]
    exclude_max_iter = viz_cfg.get("exclude_max_iteration", 20)

    def _is_excluded(name: str) -> bool:
        for r in exclude_re:
            m = r.search(name)
            if m is None:
                continue
            if m.lastindex and m.lastindex >= 1:
                if int(m.group(1)) >= exclude_max_iter:
                    continue
            return True
        return False

    if explicit:
        candidates = [models_root / name for name in explicit]
    else:
        candidates = sorted(
            p for p in models_root.iterdir() if p.is_dir() and not _is_excluded(p.name)
        )

    experiments: dict[str, Path] = {}

    for exp_dir in candidates:
        results_path = exp_dir / "evaluation" / "results.json"
        training_config_path = exp_dir / "training" / "config.yaml"

        if not results_path.exists():
            continue
        if not training_config_path.exists():
            continue

        with open(training_config_path, "r") as f:
            train_cfg = yaml.safe_load(f)

        pde_type = train_cfg.get("experiment", {}).get("pde_type")
        if pde_type != current_pde_type:
            continue

        experiments[exp_dir.name] = results_path

    # Always include current experiment
    current_results = current_exp_dir / "evaluation" / "results.json"
    if current_results.exists():
        experiments[current_exp_dir.name] = current_results

    result = sorted(experiments.items(), key=lambda x: x[0])
    print(f"  Discovered {len(result)} experiments for scatter comparison:")
    for name, _ in result:
        print(f"    - {name}")

    return result


def validate_experiment_compatibility(
    experiment_results: list[Tuple[str, dict]],
) -> Tuple[list[int], list[float]]:
    """
    Collect the union of k_values and noise_levels across all experiments.

    Each experiment may have different K values (e.g., K=800 vs K=10).
    The scatter grid uses the union as columns, and each experiment only
    populates panels matching its own K values.

    Returns the union (k_values, noise_levels), both sorted.
    """
    if not experiment_results:
        raise ValueError("No experiments to validate.")

    all_k: set[int] = set()
    all_noise: set[float] = set()

    for _name, data in experiment_results:
        all_k.update(data["config"]["k_values"])
        all_noise.update(data["config"]["noise_levels"])

    return sorted(all_k), sorted(all_noise)


def generate_cross_experiment_scatter(
    experiment_results: list[Tuple[str, dict]],
    output_dirs: list[Path],
    dpi: int,
    sel: GraphSelection = GraphSelection(
        graphs=set(), step_filters={}, experiment_filters={}, all_graphs=True
    ),
) -> None:
    """
    Generate cross-experiment coefficient recovery scatter plots.

    Generates one scatter image per fixed_step. Each panel shows true vs
    recovered coefficients across all experiments, for both MAML and baseline.

    Saves to each directory in output_dirs (supports multi-experiment placement).
    """
    k_values, noise_levels = validate_experiment_compatibility(experiment_results)

    # Discover coefficient names from first experiment (PDE-constant)
    any_task = next(iter(experiment_results[0][1]["tasks"].values()))
    specs = any_task.get("coefficient_specs", [])
    # Preserve order, deduplicate via dict
    coeff_names = list(dict.fromkeys(s["coeff_name"] for s in specs))

    if not coeff_names:
        print("  No coefficient specs found — skipping scatter.")
        return

    # Discover fixed_steps from first available combo
    scatter_fixed_steps = None
    for _name, results_data in experiment_results:
        for _tn, td in results_data["tasks"].items():
            for combo_key_str in td:
                if combo_key_str.startswith("coefficient_recovery_"):
                    fs = td[combo_key_str].get("fixed_steps")
                    if fs is not None:
                        scatter_fixed_steps = fs
                        break
            if scatter_fixed_steps is not None:
                break
        if scatter_fixed_steps is not None:
            break

    if scatter_fixed_steps is None:
        # Legacy: single-step data
        scatter_fixed_steps = [None]
        all_fixed_steps = scatter_fixed_steps
    else:
        # Keep full list for correct indexing into rec_raw arrays
        all_fixed_steps = list(scatter_fixed_steps)
        # Apply step filtering (display only selected steps)
        allowed = sel.steps_for("scatter", [int(s) for s in scatter_fixed_steps])
        scatter_fixed_steps = [s for s in scatter_fixed_steps if int(s) in allowed]

    for step_val in scatter_fixed_steps:
        step_idx = all_fixed_steps.index(step_val)
        # Build panel_data for this step
        panel_data: PanelDataDict = {}

        for coeff_name in coeff_names:
            for noise in noise_levels:
                for k in k_values:
                    key = (coeff_name, noise, k)
                    models = []

                    for dir_name, results_data in experiment_results:
                        recovery_key = f"coefficient_recovery_k_{k}_noise_{noise:.2f}"

                        # Each experiment contributes MAML + baseline entries
                        for label_suffix, model_key in [
                            ("", "maml"),
                            (" (BL)", "baseline"),
                        ]:
                            true_vals: list[float] = []
                            rec_vals: list[float] = []
                            task_names_list: list[str] = []

                            # Find step_idx for THIS experiment's fixed_steps
                            exp_fixed_steps = None
                            for _tn, _td in results_data["tasks"].items():
                                for _ck in _td:
                                    if _ck.startswith("coefficient_recovery_"):
                                        exp_fixed_steps = _td[_ck].get("fixed_steps")
                                        break
                                if exp_fixed_steps is not None:
                                    break

                            if exp_fixed_steps is None or step_val not in exp_fixed_steps:
                                continue
                            exp_step_idx = exp_fixed_steps.index(step_val)

                            for task_name, task_data in results_data["tasks"].items():
                                if recovery_key not in task_data:
                                    continue
                                model_data = task_data[recovery_key].get(model_key, {})
                                if model_data is None:
                                    continue
                                true_k = f"{coeff_name}_true"
                                rec_k = f"{coeff_name}_recovered"
                                if true_k not in model_data or rec_k not in model_data:
                                    continue

                                true_val = model_data[true_k]
                                rec_raw = model_data[rec_k]

                                # Index into step if array
                                if isinstance(rec_raw, list):
                                    if exp_step_idx >= len(rec_raw):
                                        continue
                                    rec_val = rec_raw[exp_step_idx]
                                else:
                                    rec_val = rec_raw

                                if np.isnan(true_val) or np.isnan(rec_val):
                                    continue

                                true_vals.append(true_val)
                                rec_vals.append(rec_val)
                                task_names_list.append(task_name)

                            display_name = f"{dir_name}{label_suffix}"
                            models.append(
                                (
                                    np.array(true_vals),
                                    np.array(rec_vals),
                                    task_names_list,
                                    display_name,
                                )
                            )

                    panel_data[key] = models

        filename = (
            f"step[{step_val}]_coeff_scatter_true_vs_recovered.png"
            if step_val is not None
            else "coeff_scatter_true_vs_recovered.png"
        )

        # Save to first dir via plot function, then copy to remaining dirs
        primary_path = output_dirs[0] / filename
        fig = plot_coefficient_scatter_grid(
            panel_data=panel_data,
            coeff_names=coeff_names,
            k_values=k_values,
            noise_levels=noise_levels,
            save_path=primary_path,
            dpi=dpi,
            step=step_val,
        )
        plt.close(fig)

        for out_dir in output_dirs[1:]:
            copy_path = out_dir / filename
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(primary_path, copy_path)

        print(f"  Saved scatter (step {step_val}) to {len(output_dirs)} dir(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from MAML evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML config (optional when --only scatter(exp,...) is used)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to results.json (default: auto-detect from experiment dir)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated graph names to generate (default: all). "
            "Step-filtered graphs accept [step,...] suffix. "
            "scatter accepts (exp,...) to select specific experiments. "
            "Names: " + ", ".join(sorted(VALID_GRAPH_NAMES)) + ". "
            "Example: --only scatter[1,1000](br-18,heat-1),best-combo"
        ),
    )
    args = parser.parse_args()

    # =========================================================================
    # Parse --only early (needed to decide if config is required)
    # =========================================================================
    sel = parse_only(args.only)

    print("=" * 60)
    print("MAML Visualization")
    print("=" * 60)
    print()

    # =========================================================================
    # Config-less scatter: --only scatter(...) without --config
    # =========================================================================
    exp_filter = sel.experiments_for("scatter")
    if args.config is None:
        if exp_filter is None:
            parser.error("--config is required unless --only scatter(exp,...) is used")
        non_scatter = sel.graphs - {"scatter"}
        if non_scatter:
            parser.error(
                f"--config is required for: {', '.join(sorted(non_scatter))}. "
                f"Only scatter works config-less with (exp,...) syntax."
            )

        print("-" * 60)
        print("Generating cross-experiment coefficient scatter (config-less)...")
        print("-" * 60)

        default_config: dict = {"output": {"base_dir": "data/models"}}
        discovered = resolve_experiment_names(default_config, exp_filter)
        if len(discovered) >= 1:
            experiment_results: list[Tuple[str, dict]] = []
            for dir_name, rpath in discovered:
                with open(rpath, "r") as f:
                    experiment_results.append((dir_name, json.load(f)))

            models_root = Path("data/models")
            output_dirs = [
                models_root / name / "figures" / "only" for name, _ in discovered
            ]
            generate_cross_experiment_scatter(
                experiment_results, output_dirs, 150, sel
            )
        else:
            print("  No experiments with evaluation results found.")
        print()

        print("=" * 60)
        print("Visualization complete!")
        print("=" * 60)
        return

    # =========================================================================
    # Full pipeline with --config
    # =========================================================================
    from src.config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml(args.config)
    # Keep raw dict for discover_comparison_experiments (reads other experiments' configs)
    config = cfg.to_yaml_dict()

    exp_name = cfg.experiment.name
    exp_dir = Path(cfg.output.base_dir) / exp_name

    print(f"Experiment: {exp_name}")
    print()

    dpi = cfg.visualization.dpi

    # Config-level --only (CLI already parsed above, config is fallback)
    if args.only is None:
        config_only = cfg.visualization.only
        if config_only is not None:
            sel = parse_only(config_only)

    if not exp_dir.exists():
        # Scan for suffixed directories (ISNAN → skip, ENDNAN → usable)
        candidates = sorted(exp_dir.parent.glob(f"{exp_name}-*"))
        isnan = [d for d in candidates if d.name.endswith("-ISNAN")]
        endnan = [d for d in candidates if "-ENDNAN@" in d.name]
        if isnan:
            print(f"SKIP: {isnan[0].name} is flagged ISNAN. Nothing to visualize.")
            sys.exit(0)
        elif endnan:
            exp_dir = endnan[0]
            print(f"Using flagged directory: {exp_dir.name}")
        else:
            print(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)

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
    ev = cfg.evaluation
    k_values: NDArray[np.integer[Any]] = np.array(
        results["config"].get("k_values", ev.k_values)
    )
    noise_levels: NDArray[np.floating[Any]] = np.array(
        results["config"].get("noise_levels", ev.noise_levels)
    )
    fixed_steps: NDArray[np.integer[Any]] = np.array(
        results["config"].get("fixed_steps", ev.fixed_steps)
    )
    deriv_threshold = float(ev.deriv_threshold)
    holdout_size = int(
        results["config"].get("holdout_size", ev.holdout_size)
    )
    # Compute metrics only when needed (generalization, loss-ratio, noise/sample)
    if sel.needs_metrics():
        print("-" * 60)
        print("Computing metrics...")
        print("-" * 60)

        all_metrics = compute_all_metrics(results, fixed_steps, deriv_threshold)
        print(f"Computed metrics for {len(all_metrics)} tasks")
        print()
    else:
        all_metrics = {}

    # =========================================================================
    # Cross-experiment coefficient scatter
    # =========================================================================
    if sel.enabled("scatter"):
        print("-" * 60)
        print("Generating cross-experiment coefficient scatter...")
        print("-" * 60)

        # Re-check exp_filter (sel may have been replaced by config-level --only)
        exp_filter = sel.experiments_for("scatter")
        if exp_filter is not None:
            discovered = resolve_experiment_names(config, exp_filter)
            scatter_output_dirs = [
                Path(config.get("output", {}).get("base_dir", "data/models"))
                / name
                / "figures"
                / "only"
                for name, _ in discovered
            ]
        else:
            discovered = discover_comparison_experiments(config, exp_dir)
            scatter_output_dirs = [output_dir]

        if len(discovered) >= 1:
            config_experiment_results: list[Tuple[str, dict]] = []
            for dir_name, rpath in discovered:
                with open(rpath, "r") as f:
                    config_experiment_results.append((dir_name, json.load(f)))

            generate_cross_experiment_scatter(
                config_experiment_results, scatter_output_dirs, dpi, sel
            )
        else:
            print("  No experiments with evaluation results found.")
        print()

    # Generate per-task figures
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
        sel=sel,
    )
    print()

    # Generate aggregated figures
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
        sel=sel,
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
