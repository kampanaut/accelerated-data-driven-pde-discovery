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
import re
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from numpy.typing import NDArray
import yaml
import numpy as np
import matplotlib
from src.config import ExperimentConfig, OutputSection
from src.evaluation.results import EvaluationResults, TaskResult

matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
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

    Returns GraphSelection with all_graphs=True when arg is None or empty.
    Empty string is treated as None so an unset `visualization.only: ""` in
    the config doesn't silently filter every plot out.
    """
    if arg is None or not arg.strip():
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
    # Graph 7-10: Coefficient recovery graphs
    plot_coefficient_extraction_scatter,
    plot_coefficient_heatmap,
    plot_coefficient_vs_k,
    plot_coefficient_vs_noise,
    # Graph 11: Cross-experiment scatter
    plot_coefficient_scatter_grid,
    PanelDataDict,
    # Graph 12: Best-combo prediction scatter
    plot_best_combo_scatter,
)
# `plot_jacobian_regression_scatter` is intentionally NOT imported — left
# dormant in graphs.py for any future caller that wants the (regressor, JVP)
# scatter view back.



def _loss_worse_suffix(
    kendall_worse_steps: list[int],
    mse_worse_steps: dict[str, list[int]],
    all_fixed_steps: list[int],
) -> str:
    """Build WORSE(KENDALL[...],MSE[u;...,v;...]) suffix for loss-group figures.

    `mse_worse_steps` is per-mixer: each mixer that fired contributes a
    `{mixer};{step_ranges}` segment. Includes only tags whose step lists are
    non-empty. Returns empty string if neither fired.
    """
    parts: list[str] = []
    if kendall_worse_steps:
        parts.append(f"KENDALL[{compress_step_ranges(kendall_worse_steps, all_fixed_steps)}]")
    if mse_worse_steps:
        mixer_segs = [
            f"{mname};{compress_step_ranges(steps, all_fixed_steps)}"
            for mname, steps in sorted(mse_worse_steps.items())
            if steps
        ]
        if mixer_segs:
            parts.append(f"MSE[{','.join(mixer_segs)}]")
    if not parts:
        return ""
    return f"_WORSE({','.join(parts)})"


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


def _task_dir_worse_suffix(task: TaskResult, all_fixed_steps: list[int]) -> str:
    """Build task-level directory WORSE suffix with step ranges."""
    flags = []

    if task.worse.kendall_steps:
        compressed = compress_step_ranges(task.worse.kendall_steps, all_fixed_steps)
        flags.append(f"KENDALL[{compressed}]")

    if task.worse.mse_steps:
        mixer_segs = [
            f"{mname};{compress_step_ranges(steps, all_fixed_steps)}"
            for mname, steps in sorted(task.worse.mse_steps.items())
            if steps
        ]
        if mixer_segs:
            flags.append(f"MSE[{','.join(mixer_segs)}]")

    for name, steps in task.worse.coeff_steps.items():
        if steps:
            compressed = compress_step_ranges(steps, all_fixed_steps)
            flags.append(f"COEFF:{name}[{compressed}]")

    if flags:
        return f"_WORSE({','.join(flags)})"
    return ""


def load_results_with_samples(results_path: Path) -> EvaluationResults:
    """Load evaluation results from the evaluation/ directory containing results_path."""
    return EvaluationResults.from_dir(results_path.parent)



def compute_all_metrics(
    tasks: Dict[str, TaskResult],
    fixed_steps: NDArray[np.integer[Any]],
    deriv_threshold: float = 1e-7,
) -> Dict[str, Dict[str, Dict[str, ComparisonMetrics]]]:
    """
    Compute per-mixer metrics for all (task, K, noise) combinations.

    The new schema's `per_mixer_train_losses` arrays are **sparse**: under
    L-BFGS, evaluate.py records one snapshot per fixed_step (index `i` in
    the array == loss at `fixed_steps[i]`). The classic
    `compute_comparison_metrics` from metrics.py treats `loss[step]` as a
    dense indexed lookup, which silently drops any fixed_step >= len(array).
    We bypass it here and build `ComparisonMetrics` by positional alignment.

    Plateau-related fields (`*_steps_to_plateau`, `speedup`) are degenerate
    on 2-element L-BFGS arrays — populated for shape compatibility with the
    plot functions but should be read as best-effort.

    Returns:
        Dict mapping task_name -> combo_key -> mixer_name -> ComparisonMetrics
    """
    all_metrics: Dict[str, Dict[str, Dict[str, ComparisonMetrics]]] = {}
    fs_list = [int(s) for s in fixed_steps]

    for task_name, task in tasks.items():
        all_metrics[task_name] = {}

        for combo in task.combos:
            maml_train_pm = combo.maml.fine_tune.per_mixer_train_losses
            baseline_train_pm = combo.baseline.fine_tune.per_mixer_train_losses
            maml_holdout_pm = combo.maml.fine_tune.per_mixer_holdout_losses
            baseline_holdout_pm = combo.baseline.fine_tune.per_mixer_holdout_losses

            if not maml_train_pm or not baseline_train_pm:
                continue

            per_mixer_metrics: Dict[str, ComparisonMetrics] = {}
            for mname in maml_train_pm:
                if mname not in baseline_train_pm:
                    continue

                m_train = np.asarray(maml_train_pm[mname], dtype=float)
                b_train = np.asarray(baseline_train_pm[mname], dtype=float)
                m_holdout = np.asarray(maml_holdout_pm[mname], dtype=float)
                b_holdout = np.asarray(baseline_holdout_pm[mname], dtype=float)

                # Positional alignment: array index i == fixed_steps[i].
                n = min(len(m_train), len(b_train), len(fs_list))
                if n == 0:
                    continue

                maml_at_steps: Dict[int, float] = {}
                baseline_at_steps: Dict[int, float] = {}
                ratios: Dict[int, float] = {}
                for i in range(n):
                    step = fs_list[i]
                    m_loss = float(m_train[i])
                    b_loss = float(b_train[i])
                    maml_at_steps[step] = m_loss
                    baseline_at_steps[step] = b_loss
                    ratios[step] = m_loss / b_loss if b_loss > 0 else float("inf")

                # Plateau on holdout (best-effort; degenerate for n<=2 — the
                # function returns its lower-bound default in that case).
                m_plateau_idx = (
                    steps_to_plateau(m_holdout, deriv_threshold=deriv_threshold)
                    if len(m_holdout) > 0 else 0
                )
                b_plateau_idx = (
                    steps_to_plateau(b_holdout, deriv_threshold=deriv_threshold)
                    if len(b_holdout) > 0 else 0
                )
                # Translate plateau array index → fixed_step value where possible.
                m_plateau_step = (
                    fs_list[m_plateau_idx] if m_plateau_idx < len(fs_list)
                    else fs_list[-1]
                )
                b_plateau_step = (
                    fs_list[b_plateau_idx] if b_plateau_idx < len(fs_list)
                    else fs_list[-1]
                )

                m_final = float(m_train[n - 1])
                b_final = float(b_train[n - 1])
                speedup = b_plateau_step / m_plateau_step if m_plateau_step > 0 else 1.0

                per_mixer_metrics[mname] = ComparisonMetrics(
                    maml_steps_to_plateau=m_plateau_step,
                    baseline_steps_to_plateau=b_plateau_step,
                    speedup=speedup,
                    maml_plateau_loss=m_final,
                    baseline_plateau_loss=b_final,
                    maml_loss_at_steps=maml_at_steps,
                    baseline_loss_at_steps=baseline_at_steps,
                    loss_ratios=ratios,
                )

            if per_mixer_metrics:
                all_metrics[task_name][combo.combo_key] = per_mixer_metrics

    return all_metrics


def generate_per_task_figures(
    tasks: Dict[str, TaskResult],
    all_metrics: Dict[str, Dict[str, Dict[str, ComparisonMetrics]]],
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
    """Generate all per-task figures.

    `all_metrics[task_name][combo_key][mixer_name] -> ComparisonMetrics`.
    """

    for task_name, task in tasks.items():
        suffix = _task_dir_worse_suffix(task, fixed_steps.tolist())

        task_dir = output_dir / "per_task" / f"{task_name}{suffix}"
        task_dir.mkdir(parents=True, exist_ok=True)

        task_metrics = all_metrics.get(task_name, {})
        combo_lookup = {c.combo_key: c for c in task.combos}

        # Discover coefficient names from the first combo's recovery snapshot —
        # PDE-constant across all combos.
        if not task.combos:
            continue
        coeff_names = list(
            task.combos[0].maml.coefficient_recovery.coefficients.keys()
        )

        print(f"  Generating figures for {task_name}...")

        # ---------------------------------------------------------------------
        # Graph 7: Coefficient extraction scatter (one per coeff per K × noise × step)
        # Scatter of extracted values (y) vs feature column (x) per recovery path.
        # Replaces the old histogram view with a regression-line + truth-line plot.
        # ---------------------------------------------------------------------
        if sel.enabled("jacobian"):
            allowed_steps = set(sel.steps_for("jacobian", fixed_steps.tolist()))
            step_indices = [
                i for i, s in enumerate(fixed_steps) if int(s) in allowed_steps
            ]

            for k in k_values:
                for noise in noise_levels:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo = combo_lookup[combo_key]
                    combo_coeff_worse = combo.worse.coeff_steps

                    for coeff_name in coeff_names:
                        maml_snap = combo.maml.coefficient_recovery.coefficients[coeff_name]
                        baseline_snap = combo.baseline.coefficient_recovery.coefficients[coeff_name]
                        if not maml_snap.per_step or not baseline_snap.per_step:
                            continue
                        true_val = float(maml_snap.true_value)

                        path_keys = list(maml_snap.per_step[0].recoveries.keys())

                        for si in step_indices:
                            step_label = int(fixed_steps[si])

                            maml_vals: list = []
                            maml_regs: list = []
                            baseline_vals: list = []
                            baseline_regs: list = []
                            kept_paths: list[str] = []
                            for path_key in path_keys:
                                m_v = combo.maml.per_path_raw_values.get(path_key)
                                m_r = combo.maml.per_path_regressor_values.get(path_key)
                                b_v = combo.baseline.per_path_raw_values.get(path_key)
                                b_r = combo.baseline.per_path_regressor_values.get(path_key)
                                if any(
                                    x is None for x in (m_v, m_r, b_v, b_r)
                                ) or si >= m_v.shape[0]:  # type: ignore[union-attr]
                                    continue
                                maml_vals.append(m_v[si])  # type: ignore[index]
                                maml_regs.append(m_r[si])  # type: ignore[index]
                                baseline_vals.append(b_v[si])  # type: ignore[index]
                                baseline_regs.append(b_r[si])  # type: ignore[index]
                                kept_paths.append(path_key)

                            if not kept_paths:
                                continue

                            step_suffix = f"_step[{step_label}]"
                            coeff_worse = _coeff_worse_suffix_at_step(
                                combo_coeff_worse, step_label
                            )

                            regressor_names = []
                            for path_key in kept_paths:
                                rp = maml_snap.per_step[si].recoveries.get(path_key)
                                regressor_names.append(rp.regressor_name if rp else "")

                            fig = plot_coefficient_extraction_scatter(
                                maml_values=maml_vals,
                                maml_regressors=maml_regs,
                                baseline_values=baseline_vals,
                                baseline_regressors=baseline_regs,
                                path_labels=kept_paths,
                                coeff_true=true_val,
                                title=f"{task_name}: {coeff_name} (K={k}, noise={noise:.0%}, step {step_label})",
                                coeff_name=coeff_name,
                                regressor_names=regressor_names,
                                save_path=task_dir
                                / f"coeff[{coeff_name}]_k[{k}]_noise[{noise:.2f}]{step_suffix}_distribution{coeff_worse}.png",
                                dpi=dpi,
                            )
                            plt.close(fig)

        # ---------------------------------------------------------------------
        # Generalization plot (one per K × noise) — per-mixer overlays per panel
        # ---------------------------------------------------------------------
        if sel.enabled("generalization"):
            for k in k_values:
                for noise in noise_levels:
                    combo_key = f"k_{k}_noise_{noise:.2f}"
                    combo = combo_lookup[combo_key]

                    maml_train_pm = combo.maml.fine_tune.per_mixer_train_losses
                    maml_holdout_pm = combo.maml.fine_tune.per_mixer_holdout_losses
                    baseline_train_pm = combo.baseline.fine_tune.per_mixer_train_losses
                    baseline_holdout_pm = combo.baseline.fine_tune.per_mixer_holdout_losses

                    if not maml_train_pm or not baseline_train_pm:
                        continue

                    combo_kendall_steps = combo.worse.kendall_steps
                    combo_mse_steps = combo.worse.mse_steps
                    loss_suffix = _loss_worse_suffix(
                        combo_kendall_steps, combo_mse_steps, fixed_steps.tolist()
                    )

                    fs_list = fixed_steps.tolist()

                    fig = plot_train_holdout_convergence(
                        maml_train_per_mixer=maml_train_pm,
                        maml_holdout_per_mixer=maml_holdout_pm,
                        baseline_train_per_mixer=baseline_train_pm,
                        baseline_holdout_per_mixer=baseline_holdout_pm,
                        title=f"{task_name}: K={k}, noise={noise:.0%}",
                        save_path=task_dir
                        / f"k[{k}]_noise[{noise:.2f}]_finetune_generalization{loss_suffix}.png",
                        dpi=dpi,
                        k_shot=k,
                        holdout_size=holdout_size,
                        deriv_threshold=deriv_threshold,
                        fixed_steps=fs_list,
                        kendall_worse_steps=combo_kendall_steps,
                        mse_worse_steps=combo_mse_steps,
                        per_mixer_mse_main_holdout_maml=combo.maml.per_mixer_mse_main_holdout,
                        per_mixer_aux_holdout_maml=combo.maml.per_mixer_aux_holdout,
                        per_mixer_mse_main_holdout_baseline=combo.baseline.per_mixer_mse_main_holdout,
                        per_mixer_aux_holdout_baseline=combo.baseline.per_mixer_aux_holdout,
                    )
                    plt.close(fig)

        # Discover mixer names from the first combo's fine_tune dict.
        # Used by per-mixer file splits below.
        mixer_names = list(
            task.combos[0].maml.fine_tune.per_mixer_train_losses.keys()
        )

        # ---------------------------------------------------------------------
        # Graph 4: Loss ratio heatmap (one file per mixer per fixed step)
        # ---------------------------------------------------------------------
        if sel.enabled("loss-ratio"):
            for p in sel.steps_for("loss-ratio", fixed_steps.tolist()):
                for mname in mixer_names:
                    ratios = np.full((len(noise_levels), len(k_values)), np.nan)

                    for i, noise in enumerate(noise_levels):
                        for j, k in enumerate(k_values):
                            combo_key = f"k_{k}_noise_{noise:.2f}"
                            per_mixer = task_metrics.get(combo_key, {})
                            metrics = per_mixer.get(mname)
                            if metrics is not None and p in metrics.loss_ratios:
                                ratios[i, j] = metrics.loss_ratios[p]

                    fig = plot_loss_ratio_heatmap(
                        ratios=ratios,
                        k_values=k_values,
                        noise_levels=noise_levels,
                        fixed_step=p,
                        title=f"{task_name}: Loss Ratio @ step {p} [mixer {mname}]",
                        save_path=task_dir / f"mixer[{mname}]_step[{p}]_loss_ratio_heatmap.png",
                        dpi=dpi,
                    )
                    plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 1: Noise robustness curves (one file per mixer per K)
        # ---------------------------------------------------------------------
        if sel.enabled("noise-robustness"):
            for mname in mixer_names:
                for k in k_values:
                    maml_steps_list = []
                    baseline_steps_list = []
                    valid_noise = []

                    for noise in noise_levels:
                        combo_key = f"k_{k}_noise_{noise:.2f}"
                        per_mixer = task_metrics.get(combo_key, {})
                        metrics = per_mixer.get(mname)
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
                            title=f"{task_name}: Noise Robustness (K={k}) [mixer {mname}]",
                            save_path=task_dir / f"mixer[{mname}]_k[{k}]_noise_robustness.png",
                            dpi=dpi,
                        )
                        plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 2: Sample efficiency curves (one file per mixer per noise level)
        # ---------------------------------------------------------------------
        if sel.enabled("sample-efficiency"):
            for mname in mixer_names:
                for p in fixed_steps[:1]:  # Just first fixed step
                    for noise in noise_levels:
                        maml_losses_list = []
                        baseline_losses_list = []
                        valid_k = []

                        for k in k_values:
                            combo_key = f"k_{k}_noise_{noise:.2f}"
                            per_mixer = task_metrics.get(combo_key, {})
                            metrics = per_mixer.get(mname)
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
                                title=f"{task_name}: Sample Efficiency (noise={noise:.0%}, step {p}) [mixer {mname}]",
                                save_path=task_dir
                                / f"mixer[{mname}]_noise[{noise:.2f}]_step[{p}]_sample_efficiency.png",
                                dpi=dpi,
                            )
                            plt.close(fig)

        # ---------------------------------------------------------------------
        # Graph 8: Coefficient error heatmap (one per coefficient, final step).
        # `pct_error` is precomputed per coefficient × step in the schema —
        # no more arithmetic, just read.
        # ---------------------------------------------------------------------
        if sel.enabled("coeff-heatmap"):
            for coeff_name in coeff_names:
                maml_errors = np.full((len(noise_levels), len(k_values)), np.nan)
                baseline_errors = np.full((len(noise_levels), len(k_values)), np.nan)

                for i, noise in enumerate(noise_levels):
                    for j, k in enumerate(k_values):
                        combo_key = f"k_{k}_noise_{noise:.2f}"
                        combo = combo_lookup[combo_key]

                        m_snap = combo.maml.coefficient_recovery.coefficients.get(coeff_name)
                        b_snap = combo.baseline.coefficient_recovery.coefficients.get(coeff_name)
                        if m_snap is None or not m_snap.per_step:
                            continue
                        if b_snap is None or not b_snap.per_step:
                            continue

                        maml_errors[i, j] = m_snap.per_step[-1].pct_error
                        baseline_errors[i, j] = b_snap.per_step[-1].pct_error

                if not np.all(np.isnan(maml_errors)):
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
        # Graph 9: Coefficient recovery vs K (final step)
        # ---------------------------------------------------------------------
        if sel.enabled("coeff-vs-k"):
            for coeff_name in coeff_names:
                for noise in noise_levels:
                    maml_err_list: list[float] = []
                    baseline_err_list: list[float] = []
                    valid_k: list[int] = []

                    for k in k_values:
                        combo_key = f"k_{k}_noise_{noise:.2f}"
                        combo = combo_lookup[combo_key]

                        m_snap = combo.maml.coefficient_recovery.coefficients.get(coeff_name)
                        b_snap = combo.baseline.coefficient_recovery.coefficients.get(coeff_name)
                        if m_snap is None or not m_snap.per_step:
                            continue
                        if b_snap is None or not b_snap.per_step:
                            continue

                        maml_err_list.append(m_snap.per_step[-1].pct_error)
                        baseline_err_list.append(b_snap.per_step[-1].pct_error)
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
        # Graph 10: Coefficient recovery vs noise (final step)
        # ---------------------------------------------------------------------
        if sel.enabled("coeff-vs-noise"):
            for coeff_name in coeff_names:
                for k in k_values:
                    maml_err_list = []
                    baseline_err_list = []
                    valid_noise: list[float] = []

                    for noise in noise_levels:
                        combo_key = f"k_{k}_noise_{noise:.2f}"
                        combo = combo_lookup[combo_key]

                        m_snap = combo.maml.coefficient_recovery.coefficients.get(coeff_name)
                        b_snap = combo.baseline.coefficient_recovery.coefficients.get(coeff_name)
                        if m_snap is None or not m_snap.per_step:
                            continue
                        if b_snap is None or not b_snap.per_step:
                            continue

                        maml_err_list.append(m_snap.per_step[-1].pct_error)
                        baseline_err_list.append(b_snap.per_step[-1].pct_error)
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
        bc = task.best_combo
        if bc.combo_key and bc.predictions.size > 0:
            bc_predictions = bc.predictions
            bc_true = bc.true_targets
            bc_x = bc.x_pts
            bc_y = bc.y_pts
            bc_steps = bc.steps
            bc_errors = bc.coeff_error

            if True:  # always true — data is complete when combo_key is set
                bc_key_str = bc.combo_key
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
    tasks: Dict[str, TaskResult],
    all_metrics: Dict[str, Dict[str, Dict[str, ComparisonMetrics]]],
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
    """Generate aggregated figures (mean ± std across tasks).

    `all_metrics[task_name][combo_key][mixer_name] -> ComparisonMetrics`.
    """

    # Per-task combo lookups for fast (task, combo_key) access
    task_combo_lookups = {tn: {c.combo_key: c for c in t.combos} for tn, t in tasks.items()}

    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    task_names = list(all_metrics.keys())
    n_tasks = len(task_names)

    if n_tasks == 0:
        print("  No valid results to aggregate.")
        return

    # Discover coefficient names + mixer names from the first task's first combo —
    # PDE-constant across all tasks/combos.
    any_task = next(iter(tasks.values()))
    if not any_task.combos:
        print("  No combos in first task — nothing to aggregate.")
        return
    any_combo = any_task.combos[0]
    coeff_names = list(any_combo.maml.coefficient_recovery.coefficients.keys())
    mixer_names = list(any_combo.maml.fine_tune.per_mixer_train_losses.keys())

    print(f"  Aggregating across {n_tasks} tasks...")

    # -------------------------------------------------------------------------
    # Aggregate Jacobian histograms (Graph 7).
    # Per coefficient × K × noise × step: pool per-point values across tasks
    # for each recovery path, normalised by each task's true coefficient
    # (so the histogram x-axis is "recovered / true" with truth at 1.0).
    # -------------------------------------------------------------------------
    if sel.enabled("jacobian"):
        allowed_steps = set(sel.steps_for("jacobian", fixed_steps.tolist()))
        step_indices = [
            i for i, s in enumerate(fixed_steps) if int(s) in allowed_steps
        ]

        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"

                for coeff_name in coeff_names:
                    first_tc = task_combo_lookups[task_names[0]][combo_key]
                    first_snap = first_tc.maml.coefficient_recovery.coefficients.get(coeff_name)
                    if first_snap is None or not first_snap.per_step:
                        continue
                    path_keys = list(first_snap.per_step[0].recoveries.keys())

                    for si in step_indices:
                        step_label = int(fixed_steps[si])

                        # Pool (values / true_val) and (regressors) across tasks.
                        maml_val_pool: dict[str, list[float]] = {p: [] for p in path_keys}
                        maml_reg_pool: dict[str, list[float]] = {p: [] for p in path_keys}
                        baseline_val_pool: dict[str, list[float]] = {p: [] for p in path_keys}
                        baseline_reg_pool: dict[str, list[float]] = {p: [] for p in path_keys}

                        for task_name in task_names:
                            tc = task_combo_lookups[task_name][combo_key]
                            snap = tc.maml.coefficient_recovery.coefficients.get(coeff_name)
                            if snap is None or snap.true_value == 0:
                                continue
                            true_val = float(snap.true_value)

                            for path_key in path_keys:
                                m_v = tc.maml.per_path_raw_values.get(path_key)
                                m_r = tc.maml.per_path_regressor_values.get(path_key)
                                b_v = tc.baseline.per_path_raw_values.get(path_key)
                                b_r = tc.baseline.per_path_regressor_values.get(path_key)
                                if any(x is None for x in (m_v, m_r, b_v, b_r)):
                                    continue
                                if si >= m_v.shape[0]:  # type: ignore[union-attr]
                                    continue
                                maml_val_pool[path_key].extend((m_v[si] / true_val).tolist())  # type: ignore[index]
                                maml_reg_pool[path_key].extend(m_r[si].tolist())  # type: ignore[index]
                                baseline_val_pool[path_key].extend((b_v[si] / true_val).tolist())  # type: ignore[index]
                                baseline_reg_pool[path_key].extend(b_r[si].tolist())  # type: ignore[index]

                        kept_paths = [
                            p for p in path_keys
                            if len(maml_val_pool[p]) > 0 and len(baseline_val_pool[p]) > 0
                        ]
                        if not kept_paths:
                            continue

                        step_suffix = f"_step[{step_label}]"
                        agg_regressor_names = []
                        first_tc_snap = task_combo_lookups[task_names[0]][combo_key]
                        first_agg_snap = first_tc_snap.maml.coefficient_recovery.coefficients.get(coeff_name)
                        for path_key in kept_paths:
                            rp = (
                                first_agg_snap.per_step[si].recoveries.get(path_key)
                                if first_agg_snap and si < len(first_agg_snap.per_step)
                                else None
                            )
                            agg_regressor_names.append(rp.regressor_name if rp else "")
                        fig = plot_coefficient_extraction_scatter(
                            maml_values=[np.array(maml_val_pool[p]) for p in kept_paths],
                            maml_regressors=[np.array(maml_reg_pool[p]) for p in kept_paths],
                            baseline_values=[np.array(baseline_val_pool[p]) for p in kept_paths],
                            baseline_regressors=[np.array(baseline_reg_pool[p]) for p in kept_paths],
                            path_labels=kept_paths,
                            coeff_true=1.0,
                            title=f"Aggregated {coeff_name} Recovery Ratio (K={k}, noise={noise:.0%}, step {step_label}, n={n_tasks} tasks)",
                            coeff_name=coeff_name,
                            regressor_names=agg_regressor_names,
                            save_path=agg_dir
                            / f"coeff[{coeff_name}]_k[{k}]_noise[{noise:.2f}]{step_suffix}_distribution.png",
                            dpi=dpi,
                            ratio_mode=True,
                        )
                        plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate loss ratio heatmaps (one file per mixer per fixed step)
    # -------------------------------------------------------------------------
    if sel.enabled("loss-ratio"):
        for p in sel.steps_for("loss-ratio", fixed_steps.tolist()):
            for mname in mixer_names:
                ratio_stack = []
                for task_name in task_names:
                    task_metrics = all_metrics[task_name]
                    ratios = np.full((len(noise_levels), len(k_values)), np.nan)

                    for i, noise in enumerate(noise_levels):
                        for j, k in enumerate(k_values):
                            combo_key = f"k_{k}_noise_{noise:.2f}"
                            per_mixer = task_metrics.get(combo_key, {})
                            metrics = per_mixer.get(mname)
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
                    title=f"Aggregated Loss Ratio @ step {p} (n={n_tasks} tasks) [mixer {mname}]",
                    save_path=agg_dir / f"mixer[{mname}]_step[{p}]_loss_ratio_heatmap.png",
                    dpi=dpi,
                    std_values=ratio_std,
                    inf_counts=ratio_inf_counts,
                    n_total=n_tasks,
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate generalization plots (one per K × noise) — per-mixer overlays
    # -------------------------------------------------------------------------
    if sel.enabled("generalization"):
        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"

                # Collect per-mixer curves across all tasks for this combo.
                # Outer dict keyed by mixer name; values are lists of per-task curves.
                maml_train_acc: Dict[str, list] = {}
                maml_holdout_acc: Dict[str, list] = {}
                baseline_train_acc: Dict[str, list] = {}
                baseline_holdout_acc: Dict[str, list] = {}
                maml_plateau_acc: Dict[str, list[int]] = {}
                baseline_plateau_acc: Dict[str, list[int]] = {}

                for task_name in task_names:
                    tc = task_combo_lookups[task_name][combo_key]
                    mt_pm = tc.maml.fine_tune.per_mixer_train_losses
                    mh_pm = tc.maml.fine_tune.per_mixer_holdout_losses
                    bt_pm = tc.baseline.fine_tune.per_mixer_train_losses
                    bh_pm = tc.baseline.fine_tune.per_mixer_holdout_losses

                    if not mt_pm or not bt_pm:
                        continue

                    for mname, mt in mt_pm.items():
                        mh = mh_pm[mname]
                        maml_train_acc.setdefault(mname, []).append(mt)
                        maml_holdout_acc.setdefault(mname, []).append(mh)
                        maml_plateau_acc.setdefault(mname, []).append(
                            steps_to_plateau(mh)
                        )
                    for mname, bt in bt_pm.items():
                        bh = bh_pm[mname]
                        baseline_train_acc.setdefault(mname, []).append(bt)
                        baseline_holdout_acc.setdefault(mname, []).append(bh)
                        baseline_plateau_acc.setdefault(mname, []).append(
                            steps_to_plateau(bh)
                        )

                if not maml_train_acc or not baseline_train_acc:
                    continue

                # Per-mixer mean / std across tasks. Truncate to min_len per mixer
                # in case fixed_steps differs across tasks (shouldn't, but safe).
                def _mean_std(
                    acc: Dict[str, list]
                ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
                    means: Dict[str, NDArray] = {}
                    stds: Dict[str, NDArray] = {}
                    for mname, curves in acc.items():
                        min_len = min(len(c) for c in curves)
                        stack = np.array([c[:min_len] for c in curves])
                        means[mname] = np.mean(stack, axis=0)
                        stds[mname] = np.std(stack, axis=0)
                    return means, stds

                maml_train_mean, maml_train_std = _mean_std(maml_train_acc)
                maml_holdout_mean, maml_holdout_std = _mean_std(maml_holdout_acc)
                baseline_train_mean, baseline_train_std = _mean_std(baseline_train_acc)
                baseline_holdout_mean, baseline_holdout_std = _mean_std(baseline_holdout_acc)

                maml_plateau_mean = {
                    m: int(np.mean(v)) for m, v in maml_plateau_acc.items()
                }
                baseline_plateau_mean = {
                    m: int(np.mean(v)) for m, v in baseline_plateau_acc.items()
                }
                maml_plateau_std = {
                    m: float(np.std(v)) for m, v in maml_plateau_acc.items()
                }
                baseline_plateau_std = {
                    m: float(np.std(v)) for m, v in baseline_plateau_acc.items()
                }

                n_tasks_used = len(next(iter(maml_train_acc.values())))

                fig = plot_train_holdout_convergence(
                    maml_train_per_mixer=maml_train_mean,
                    maml_holdout_per_mixer=maml_holdout_mean,
                    baseline_train_per_mixer=baseline_train_mean,
                    baseline_holdout_per_mixer=baseline_holdout_mean,
                    title=f"Aggregated: K={k}, noise={noise:.0%} (n={n_tasks_used} tasks)",
                    save_path=agg_dir
                    / f"k[{k}]_noise[{noise:.2f}]_finetune_generalization.png",
                    dpi=dpi,
                    k_shot=k,
                    holdout_size=holdout_size,
                    maml_train_std_per_mixer=maml_train_std,
                    maml_holdout_std_per_mixer=maml_holdout_std,
                    baseline_train_std_per_mixer=baseline_train_std,
                    baseline_holdout_std_per_mixer=baseline_holdout_std,
                    deriv_threshold=deriv_threshold,
                    maml_plateau_steps_per_mixer=maml_plateau_mean,
                    baseline_plateau_steps_per_mixer=baseline_plateau_mean,
                    maml_plateau_step_std_per_mixer=maml_plateau_std,
                    baseline_plateau_step_std_per_mixer=baseline_plateau_std,
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate noise robustness curves (one file per mixer per K)
    # -------------------------------------------------------------------------
    if sel.enabled("noise-robustness"):
        for mname in mixer_names:
            for k in k_values:
                maml_steps_per_task = []
                baseline_steps_per_task = []

                for task_name in task_names:
                    task_metrics = all_metrics[task_name]
                    maml_row = []
                    baseline_row = []

                    for noise in noise_levels:
                        combo_key = f"k_{k}_noise_{noise:.2f}"
                        per_mixer = task_metrics.get(combo_key, {})
                        metrics = per_mixer.get(mname)

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
                    title=f"Aggregated Noise Robustness (K={k}, n={n_tasks} tasks) [mixer {mname}]",
                    save_path=agg_dir / f"mixer[{mname}]_k[{k}]_noise_robustness.png",
                    dpi=dpi,
                    maml_std=np.nanstd(maml_arr, axis=0),
                    baseline_std=np.nanstd(baseline_arr, axis=0),
                )
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate sample efficiency curves (one file per mixer per noise)
    # -------------------------------------------------------------------------
    if sel.enabled("sample-efficiency"):
        for mname in mixer_names:
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
                            per_mixer = task_metrics.get(combo_key, {})
                            metrics = per_mixer.get(mname)

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
                        title=f"Aggregated Sample Efficiency (noise={noise:.0%}, step {p}, n={n_tasks} tasks) [mixer {mname}]",
                        save_path=agg_dir
                        / f"mixer[{mname}]_noise[{noise:.2f}]_step[{p}]_sample_efficiency.png",
                        dpi=dpi,
                        maml_std=np.nanstd(maml_arr, axis=0),
                        baseline_std=np.nanstd(baseline_arr, axis=0),
                    )
                    plt.close(fig)

    # -------------------------------------------------------------------------
    # Aggregate coefficient error heatmap — read pct_error per task per (k, noise),
    # mean across tasks per cell.
    # -------------------------------------------------------------------------
    if sel.enabled("coeff-heatmap"):
        for coeff_name in coeff_names:
            maml_error_stack = []
            baseline_error_stack = []

            for task_name in task_names:
                tcl = task_combo_lookups[task_name]
                maml_errors = np.full((len(noise_levels), len(k_values)), np.nan)
                baseline_errors = np.full((len(noise_levels), len(k_values)), np.nan)

                for i, noise in enumerate(noise_levels):
                    for j, k in enumerate(k_values):
                        combo = tcl[f"k_{k}_noise_{noise:.2f}"]
                        m_snap = combo.maml.coefficient_recovery.coefficients.get(coeff_name)
                        b_snap = combo.baseline.coefficient_recovery.coefficients.get(coeff_name)
                        if m_snap is None or not m_snap.per_step:
                            continue
                        if b_snap is None or not b_snap.per_step:
                            continue
                        maml_errors[i, j] = m_snap.per_step[-1].pct_error
                        baseline_errors[i, j] = b_snap.per_step[-1].pct_error

                maml_error_stack.append(maml_errors)
                baseline_error_stack.append(baseline_errors)

            maml_error_mean = np.nanmean(np.array(maml_error_stack), axis=0)
            baseline_error_mean = np.nanmean(np.array(baseline_error_stack), axis=0)

            if not np.all(np.isnan(maml_error_mean)):
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
    # Aggregate coefficient vs K (final step, mean across tasks)
    # -------------------------------------------------------------------------
    if sel.enabled("coeff-vs-k"):
        for coeff_name in coeff_names:
            for noise in noise_levels:
                maml_per_task = []
                baseline_per_task = []

                for task_name in task_names:
                    tcl = task_combo_lookups[task_name]
                    maml_row: list[float] = []
                    baseline_row: list[float] = []

                    for k in k_values:
                        combo = tcl[f"k_{k}_noise_{noise:.2f}"]
                        m_snap = combo.maml.coefficient_recovery.coefficients.get(coeff_name)
                        b_snap = combo.baseline.coefficient_recovery.coefficients.get(coeff_name)
                        if (
                            m_snap is None or not m_snap.per_step
                            or b_snap is None or not b_snap.per_step
                        ):
                            maml_row.append(np.nan)
                            baseline_row.append(np.nan)
                        else:
                            maml_row.append(m_snap.per_step[-1].pct_error)
                            baseline_row.append(b_snap.per_step[-1].pct_error)

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
    # Aggregate coefficient vs noise (final step, mean across tasks)
    # -------------------------------------------------------------------------
    if sel.enabled("coeff-vs-noise"):
        for coeff_name in coeff_names:
            for k in k_values:
                maml_per_task = []
                baseline_per_task = []

                for task_name in task_names:
                    tcl = task_combo_lookups[task_name]
                    maml_row = []
                    baseline_row = []

                    for noise in noise_levels:
                        combo = tcl[f"k_{k}_noise_{noise:.2f}"]
                        m_snap = combo.maml.coefficient_recovery.coefficients.get(coeff_name)
                        b_snap = combo.baseline.coefficient_recovery.coefficients.get(coeff_name)
                        if (
                            m_snap is None or not m_snap.per_step
                            or b_snap is None or not b_snap.per_step
                        ):
                            maml_row.append(np.nan)
                            baseline_row.append(np.nan)
                        else:
                            maml_row.append(m_snap.per_step[-1].pct_error)
                            baseline_row.append(b_snap.per_step[-1].pct_error)

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
    cfg: "ExperimentConfig", experiment_names: list[str]
) -> list[Tuple[str, Path]]:
    """
    Resolve CLI-provided experiment names to (name, results_path) pairs.

    Args:
        cfg: Experiment config (for base_dir)
        experiment_names: List of experiment directory names from --only scatter(...)

    Returns:
        List of (dir_name, results_json_path) sorted by dir_name
    """
    models_root = Path(cfg.output.base_dir)

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
    cfg: "ExperimentConfig", current_exp_dir: Path
) -> list[Tuple[str, Path]]:
    """
    Discover experiments to compare in the coefficient scatter plot.

    If visualization.compare_experiments is set, use those dir names.
    Otherwise auto-discover all data/models/* with matching pde_type
    and an existing evaluation/results.json.

    Returns list of (dir_name, results_json_path) sorted by dir_name.
    """
    current_pde_type = cfg.experiment.pde_type
    models_root = Path(cfg.output.base_dir)

    viz = cfg.visualization
    explicit = viz.compare_experiments

    # Suffix exclusion: regexes tested against dir name. Capture group = iteration count.
    exclude_patterns = [r"-ISNAN$", r"-ENDNAN@(\d+)$"] + viz.exclude_suffixes_append
    exclude_re = [re.compile(p) for p in exclude_patterns]
    exclude_max_iter = viz.exclude_max_iteration

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
    experiment_results: list[Tuple[str, EvaluationResults]],
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

    for _name, er in experiment_results:
        all_k.update(er.config.k_values)
        all_noise.update(er.config.noise_levels)

    return sorted(all_k), sorted(all_noise)


def generate_cross_experiment_scatter(
    experiment_results: list[Tuple[str, EvaluationResults]],
    output_dirs: list[Path],
    dpi: int,
    sel: GraphSelection = GraphSelection(
        graphs=set(), step_filters={}, experiment_filters={}, all_graphs=True
    ),
    train_coeff_values: Optional[dict[str, list[float]]] = None,
) -> None:
    """
    Generate cross-experiment coefficient recovery scatter plots.

    One file per coefficient per step. Each model entry inside the panel is
    keyed `(experiment, method, path_key)` so multi-path coefficients (BR's
    k2, λω's c) appear with one scatter series per recovery path, marker
    cycled by MODEL_MARKERS.
    """
    k_values, noise_levels = validate_experiment_compatibility(experiment_results)

    # Discover coefficient names from the first experiment's first task (PDE-constant).
    first_er = experiment_results[0][1]
    any_task = next(iter(first_er.tasks.values()), None)
    if any_task is None or not any_task.combos:
        print("  No tasks/combos in first experiment — skipping scatter.")
        return
    coeff_names = list(
        any_task.combos[0].maml.coefficient_recovery.coefficients.keys()
    )
    if not coeff_names:
        print("  No coefficients found — skipping scatter.")
        return

    # Discover fixed_steps from the first experiment with a non-empty per_step.
    scatter_fixed_steps: Optional[list[int]] = None
    for _name, er in experiment_results:
        for task in er.tasks.values():
            if not task.combos:
                continue
            cr = task.combos[0].maml.coefficient_recovery
            for snap in cr.coefficients.values():
                if snap.per_step:
                    scatter_fixed_steps = list(er.config.fixed_steps)
                    break
            if scatter_fixed_steps:
                break
        if scatter_fixed_steps:
            break

    if not scatter_fixed_steps:
        print("  No fixed_steps with recovery data — skipping scatter.")
        return

    allowed = set(sel.steps_for("scatter", [int(s) for s in scatter_fixed_steps]))
    scatter_fixed_steps = [s for s in scatter_fixed_steps if int(s) in allowed]

    for step_val in scatter_fixed_steps:
        for coeff_name in coeff_names:
            # Discover path keys for this coefficient from the first task that has it.
            path_keys_for_coeff: list[str] = []
            for _name, er in experiment_results:
                for task in er.tasks.values():
                    if not task.combos:
                        continue
                    snap = task.combos[0].maml.coefficient_recovery.coefficients.get(coeff_name)
                    if snap and snap.per_step:
                        path_keys_for_coeff = list(snap.per_step[0].recoveries.keys())
                        break
                if path_keys_for_coeff:
                    break

            if not path_keys_for_coeff:
                continue

            panel_data: PanelDataDict = {}

            for noise in noise_levels:
                for k in k_values:
                    key = (coeff_name, noise, k)
                    models = []

                    combo_key = f"k_{k}_noise_{noise:.2f}"

                    for dir_name, er in experiment_results:
                        exp_fs = er.config.fixed_steps
                        if not exp_fs or step_val not in exp_fs:
                            continue
                        exp_step_idx = exp_fs.index(step_val)

                        for method_label, get_method in [
                            ("", lambda c: c.maml),
                            (" (BL)", lambda c: c.baseline),
                        ]:
                            for path_key in path_keys_for_coeff:
                                true_vals: list[float] = []
                                rec_vals: list[float] = []
                                task_names_list: list[str] = []

                                for task_name, task in er.tasks.items():
                                    try:
                                        combo = task.combo_by_key(combo_key)
                                    except KeyError:
                                        continue

                                    method = get_method(combo)
                                    snap = method.coefficient_recovery.coefficients.get(coeff_name)
                                    if snap is None or exp_step_idx >= len(snap.per_step):
                                        continue
                                    ps = snap.per_step[exp_step_idx]
                                    rp = ps.recoveries.get(path_key)
                                    if rp is None:
                                        continue

                                    true_val = float(snap.true_value)
                                    rec_val = float(rp.mean)

                                    if np.isnan(true_val) or np.isnan(rec_val):
                                        continue

                                    true_vals.append(true_val)
                                    rec_vals.append(rec_val)
                                    task_names_list.append(task_name)

                                if not true_vals:
                                    continue

                                display_name = f"{dir_name}/{path_key}{method_label}"
                                models.append(
                                    (
                                        np.array(true_vals),
                                        np.array(rec_vals),
                                        task_names_list,
                                        display_name,
                                    )
                                )

                    panel_data[key] = models

            filename = f"{coeff_name}.step[{step_val}]_coeff_scatter_true_vs_recovered.png"

            primary_path = output_dirs[0] / filename
            fig = plot_coefficient_scatter_grid(
                panel_data=panel_data,
                coeff_names=[coeff_name],
                k_values=k_values,
                noise_levels=noise_levels,
                save_path=primary_path,
                dpi=dpi,
                step=step_val,
                train_coeff_values=train_coeff_values,
            )
            plt.close(fig)

            for out_dir in output_dirs[1:]:
                copy_path = out_dir / filename
                copy_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(primary_path, copy_path)

            print(f"  Saved {coeff_name} scatter (step {step_val}) to {len(output_dirs)} dir(s)")


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

        default_cfg = ExperimentConfig(output=OutputSection(base_dir="data/models"))
        discovered = resolve_experiment_names(default_cfg, exp_filter)
        if len(discovered) >= 1:
            experiment_results: list[Tuple[str, EvaluationResults]] = []
            for dir_name, rpath in discovered:
                experiment_results.append(
                    (dir_name, EvaluationResults.from_dir(rpath.parent))
                )

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
    cfg = ExperimentConfig.from_yaml(args.config)
    # cfg is now passed directly to discover/resolve functions — no raw dict needed

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
    print(f"Tasks: {len(results.tasks)}")
    print()

    # Get parameters — prefer results.json config (what evaluation actually used)
    rc = results.config
    ev = cfg.evaluation
    k_values: NDArray[np.integer[Any]] = np.array(rc.k_values or ev.k_values)
    noise_levels: NDArray[np.floating[Any]] = np.array(rc.noise_levels or ev.noise_levels)
    fixed_steps: NDArray[np.integer[Any]] = np.array(rc.fixed_steps or ev.fixed_steps)
    deriv_threshold = float(ev.deriv_threshold)
    holdout_size = int(rc.holdout_size or ev.holdout_size)
    # Compute metrics only when needed (generalization, loss-ratio, noise/sample)
    if sel.needs_metrics():
        print("-" * 60)
        print("Computing metrics...")
        print("-" * 60)

        all_metrics = compute_all_metrics(results.tasks, fixed_steps, deriv_threshold)
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
            discovered = resolve_experiment_names(cfg, exp_filter)
            scatter_output_dirs = [
                Path(cfg.output.base_dir)
                / name
                / "figures"
                / "only"
                for name, _ in discovered
            ]
        else:
            discovered = discover_comparison_experiments(cfg, exp_dir)
            scatter_output_dirs = [output_dir]

        if len(discovered) >= 1:
            config_experiment_results: list[Tuple[str, EvaluationResults]] = []
            for dir_name, rpath in discovered:
                config_experiment_results.append(
                    (dir_name, EvaluationResults.from_dir(rpath.parent))
                )

            # Load training coefficient values for distribution lines
            train_coeffs: Optional[dict[str, list[float]]] = None
            train_dir = Path(cfg.data.meta_train_dir)
            if train_dir.exists():
                train_coeffs = {}
                for npz_path in sorted(train_dir.glob("*.npz")):
                    sp = np.load(npz_path, allow_pickle=True)["simulation_params"].item()
                    for k, v in sp.items():
                        if isinstance(v, (int, float)):
                            train_coeffs.setdefault(k, []).append(float(v))

            generate_cross_experiment_scatter(
                config_experiment_results, scatter_output_dirs, dpi, sel,
                train_coeff_values=train_coeffs,
            )
        else:
            print("  No experiments with evaluation results found.")
        print()

    # Generate per-task figures
    print("-" * 60)
    print("Generating per-task figures...")
    print("-" * 60)
    generate_per_task_figures(
        tasks=results.tasks,
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
        tasks=results.tasks,
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
