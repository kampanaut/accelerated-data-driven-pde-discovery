"""
Visualization functions for MAML experiment results.

Graph types from experiment_bible.md:
- Graph 1: Noise robustness curves (steps to threshold vs noise level)
- Graph 2: Sample efficiency curves (loss at step p vs K)
- Graph 3: Speedup heatmap (K × Noise grid)
- Graph 4: Loss ratio heatmap (K × Noise grid)
- Graph 5: Convergence plot (loss vs steps)
- Graph 6: Train-holdout convergence (train vs holdout loss)
- Graph 7: Jacobian histogram (coefficient distribution)
- Graph 8: Coefficient error heatmap (K × Noise grid)
- Graph 9: Coefficient vs K (recovery vs sample size)
- Graph 10: Coefficient vs Noise (recovery vs noise level)

Each function supports both single-task and aggregated (mean ± std) modes.
"""

from typing import Any, Optional, Tuple
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.figure as pltf
import matplotlib.axes as pltax
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
from scipy import stats

from src.evaluation.metrics import steps_to_plateau

# Cross-experiment scatter plot constants
# IC colors: assigned dynamically from this palette, sorted alphabetically by IC type
IC_PALETTE: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
    "#7f7f7f",
    "#aec7e8",
    "#ffbb78",
]
MODEL_MARKERS: list[str] = [
    "o",
    "^",
    "*",
    "D",
    "v",
    "s",
    "P",
    "X",
    "h",
    "<",
    ">",
    "d",
    "p",
    "H",
    "8",
    "1",
]
# MAML = dark (saturated), baseline = light (desaturated) variant of same hue
MODEL_COLORS_DARK: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
]
MODEL_COLORS_LIGHT: list[str] = [
    "#72b7e6",
    "#ffb87a",
    "#7ada7a",
    "#e98787",
    "#c4abda",
    "#c69d95",
    "#efb4dd",
    "#b8b8b8",
    "#e6e679",
    "#74e4ef",
    "#d2e0f2",
    "#ffd9b4",
    "#c6edbe",
    "#fec6c5",
    "#dfd3e7",
    "#dec8c4",
]


def plot_train_holdout_convergence(
    maml_train: NDArray[np.floating[Any]],
    maml_holdout: NDArray[np.floating[Any]],
    baseline_train: NDArray[np.floating[Any]],
    baseline_holdout: NDArray[np.floating[Any]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    k_shot: Optional[int] = None,
    holdout_size: Optional[int] = None,
    maml_train_std: Optional[NDArray[np.floating[Any]]] = None,
    maml_holdout_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_train_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_holdout_std: Optional[NDArray[np.floating[Any]]] = None,
    deriv_threshold: float = 1e-7,
    maml_plateau_step: Optional[int] = None,
    baseline_plateau_step: Optional[int] = None,
    plateau_step_std: Optional[Tuple[float, float]] = None,
    fixed_steps: Optional[list[int]] = None,
    loss_worse_steps: Optional[list[int]] = None,
) -> pltf.Figure:
    """
    Generalization plot: train vs holdout loss with plateau and step markers.

    Shows generalization gap, plateau detection, and per-step MAML vs baseline
    comparison via color-coded vertical lines at each fixed_step.

    Args:
        maml_train: MAML training loss curve (or mean if aggregated)
        maml_holdout: MAML holdout loss curve (or mean if aggregated)
        baseline_train: Baseline training loss curve (or mean if aggregated)
        baseline_holdout: Baseline holdout loss curve (or mean if aggregated)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        k_shot: Number of support samples (for labels)
        holdout_size: Number of holdout samples (for labels)
        maml_train_std: Std for MAML train losses (aggregated mode)
        maml_holdout_std: Std for MAML holdout losses (aggregated mode)
        baseline_train_std: Std for baseline train losses (aggregated mode)
        baseline_holdout_std: Std for baseline holdout losses (aggregated mode)
        deriv_threshold: Threshold for plateau detection
        maml_plateau_step: Pre-computed MAML plateau step (None = auto-detect)
        baseline_plateau_step: Pre-computed baseline plateau step
        plateau_step_std: (maml_std, baseline_std) for plateau step bands
        fixed_steps: Steps at which Jacobian was extracted (for markers)
        loss_worse_steps: Subset of fixed_steps where MAML holdout > baseline

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    steps = np.arange(len(maml_train))

    # Compute overall minimum holdout loss for reference line
    min_holdout_loss = min(maml_holdout.min(), baseline_holdout.min())

    # Build legend labels with sample sizes
    train_label = f"Train (K={k_shot})" if k_shot is not None else "Train"
    holdout_label = (
        f"Holdout (n={holdout_size})" if holdout_size is not None else "Holdout"
    )

    # --- Fixed step markers (color-coded, drawn first so curves sit on top) ---
    if fixed_steps is not None:
        worse_set = set(loss_worse_steps) if loss_worse_steps is not None else set()
        for s in fixed_steps:
            if s > len(maml_train):
                continue
            color = "#e74c3c" if s in worse_set else "#2ecc71"
            alpha = 0.35 if s in worse_set else 0.2
            for ax in (ax1, ax2):
                ax.axvline(x=s, color=color, linewidth=1.0, alpha=alpha)

    # MAML subplot
    ax1.semilogy(steps, maml_train, "b-", label=train_label, linewidth=2)
    ax1.semilogy(
        steps, maml_holdout, "b--", label=holdout_label, linewidth=2, alpha=0.7
    )

    # Add std bands for MAML (aggregated mode)
    if maml_train_std is not None:
        ax1.fill_between(
            steps,
            maml_train - maml_train_std,
            maml_train + maml_train_std,
            color="blue",
            alpha=0.2,
        )
    if maml_holdout_std is not None:
        ax1.fill_between(
            steps,
            maml_holdout - maml_holdout_std,
            maml_holdout + maml_holdout_std,
            color="blue",
            alpha=0.1,
        )

    # Baseline subplot
    ax2.semilogy(steps, baseline_train, "r-", label=train_label, linewidth=2)
    ax2.semilogy(
        steps, baseline_holdout, "r--", label=holdout_label, linewidth=2, alpha=0.7
    )

    # Add std bands for baseline (aggregated mode)
    if baseline_train_std is not None:
        ax2.fill_between(
            steps,
            baseline_train - baseline_train_std,
            baseline_train + baseline_train_std,
            color="red",
            alpha=0.2,
        )
    if baseline_holdout_std is not None:
        ax2.fill_between(
            steps,
            baseline_holdout - baseline_holdout_std,
            baseline_holdout + baseline_holdout_std,
            color="red",
            alpha=0.1,
        )

    # --- Plateau detection on holdout curves ---
    if maml_plateau_step is None:
        maml_plateau_step = steps_to_plateau(
            maml_holdout, deriv_threshold=deriv_threshold
        )
    if baseline_plateau_step is None:
        baseline_plateau_step = steps_to_plateau(
            baseline_holdout, deriv_threshold=deriv_threshold
        )

    # Mark plateau on respective subplots
    ax1.plot(
        maml_plateau_step,
        maml_holdout[maml_plateau_step],
        "bo",
        markersize=8,
        label=f"Plateau @ {maml_plateau_step}",
        zorder=5,
    )
    ax1.axvline(x=maml_plateau_step, color="blue", linestyle=":", alpha=0.5)

    ax2.plot(
        baseline_plateau_step,
        baseline_holdout[baseline_plateau_step],
        "rs",
        markersize=8,
        label=f"Plateau @ {baseline_plateau_step}",
        zorder=5,
    )
    ax2.axvline(x=baseline_plateau_step, color="red", linestyle=":", alpha=0.5)

    # Plateau std bands (aggregated mode)
    if plateau_step_std is not None:
        maml_step_std, baseline_step_std = plateau_step_std
        ax1.axvspan(
            maml_plateau_step - maml_step_std,
            maml_plateau_step + maml_step_std,
            color="blue",
            alpha=0.1,
        )
        ax2.axvspan(
            baseline_plateau_step - baseline_step_std,
            baseline_plateau_step + baseline_step_std,
            color="red",
            alpha=0.1,
        )

    # --- Labels and formatting ---
    ax1.set_xlabel("Gradient Steps")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("MAML (θ*)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Gradient Steps")
    ax2.set_title("Baseline (θ₀)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Draw thin horizontal line at overall minimum holdout loss
    min_line_color = "gray"
    min_line_alpha = 0.7
    min_linewidth = 0.8
    ax1.axhline(
        y=min_holdout_loss,
        color=min_line_color,
        linewidth=min_linewidth,
        alpha=min_line_alpha,
    )
    ax2.axhline(
        y=min_holdout_loss,
        color=min_line_color,
        linewidth=min_linewidth,
        alpha=min_line_alpha,
    )
    ax2.annotate(
        f"min: {min_holdout_loss:.2e}",
        xy=(steps[-1], min_holdout_loss),
        fontsize=7,
        color=min_line_color,
        alpha=min_line_alpha,
        va="bottom",
        ha="right",
    )

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig



def plot_loss_ratio_heatmap(
    ratios: NDArray[np.floating[Any]],
    k_values: NDArray[np.integer[Any]],
    noise_levels: NDArray[np.floating[Any]],
    fixed_step: int,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    std_values: Optional[NDArray[np.floating[Any]]] = None,
    inf_counts: Optional[NDArray[np.floating[Any]]] = None,
    n_total: Optional[int] = None,
) -> pltf.Figure:
    """
    Graph 4: Loss ratio heatmap (K × Noise) at fixed step p.

    Cell value = maml_loss / baseline_loss. Values < 1 mean MAML wins.

    Args:
        ratios: 2D array of loss ratios, shape (n_noise, n_K)
        k_values: K values for x-axis labels
        noise_levels: Noise levels for y-axis labels
        fixed_step: The step p at which ratios were computed
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        std_values: Optional std array for annotations (aggregated mode)
        inf_counts: Optional 2D array of inf counts per cell (aggregated mode)
        n_total: Total number of tasks aggregated (for showing inf ratio)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.isnan(ratios)

    # Diverging colormap centered at 1.0
    im = ax.imshow(ratios, cmap="RdYlGn_r", aspect="auto", vmin=0.2, vmax=1.8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Loss Ratio @ step {fixed_step} (<1 = MAML better)")

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(noise_levels)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([f"{n:.0%}" for n in noise_levels])
    ax.set_xlabel("K (support set size)")
    ax.set_ylabel("Noise Level")

    has_inf_counts = inf_counts is not None

    for i in range(len(noise_levels)):
        for j in range(len(k_values)):
            if mask[i, j]:
                text = "N/A"
                color = "gray"
            else:
                value = ratios[i, j]
                n_inf = int(inf_counts[i, j]) if has_inf_counts else 0

                # Aggregated mode with some inf values
                if has_inf_counts and n_inf > 0:
                    # Draw turquoise background
                    rect = Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="#00CED1", edgecolor="none"
                    )
                    ax.add_patch(rect)
                    # Show inf count + finite stats if available
                    if not np.isnan(value) and std_values is not None:
                        text = f"∞ ({n_inf}/{n_total})\nμ={value:.5f} ±{std_values[i, j]:.5f}"
                    else:
                        text = f"∞ ({n_inf}/{n_total})"
                    color = "white"
                # Single-task mode with inf value
                elif np.isinf(value):
                    # Draw turquoise background for inf cells
                    rect = Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="#00CED1", edgecolor="none"
                    )
                    ax.add_patch(rect)
                    text = "∞"
                    color = "white"
                elif std_values is not None:
                    text = f"{value:.5f}\n±{std_values[i, j]:.5f}"
                    color = "black" if 0.4 < value < 1.6 else "white"
                else:
                    text = f"{value:.5f}"
                    color = "black" if 0.4 < value < 1.6 else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_noise_robustness(
    noise_levels: NDArray[np.floating[Any]],
    maml_steps: NDArray[np.floating[Any]],
    baseline_steps: NDArray[np.floating[Any]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_std: Optional[NDArray[np.floating[Any]]] = None,
) -> pltf.Figure:
    """
    Graph 1: Noise robustness curves.

    Shows how noise affects time to reach plateau (steps_to_plateau).

    Args:
        noise_levels: X-axis values (noise percentages as decimals, e.g., 0.01 for 1%)
        maml_steps: Steps to plateau for MAML at each noise level
        baseline_steps: Steps to plateau for baseline at each noise level
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        maml_std: Standard deviation for MAML (aggregated mode)
        baseline_std: Standard deviation for baseline (aggregated mode)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    noise_pct = [n * 100 for n in noise_levels]

    ax.plot(noise_pct, maml_steps, "b-o", label="MAML (θ*)", linewidth=2, markersize=8)
    ax.plot(
        noise_pct,
        baseline_steps,
        "r-s",
        label="Baseline (θ₀)",
        linewidth=2,
        markersize=8,
    )

    if maml_std is not None:
        ax.fill_between(
            noise_pct,
            maml_steps - maml_std,
            maml_steps + maml_std,
            color="blue",
            alpha=0.2,
        )
    if baseline_std is not None:
        ax.fill_between(
            noise_pct,
            baseline_steps - baseline_std,
            baseline_steps + baseline_std,
            color="red",
            alpha=0.2,
        )

    ax.set_xlabel("Noise Level (%)")
    ax.set_ylabel("Steps to Plateau")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_sample_efficiency(
    k_values: NDArray[np.integer[Any]],
    maml_losses: NDArray[np.floating[Any]],
    baseline_losses: NDArray[np.floating[Any]],
    fixed_step: int,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_std: Optional[NDArray[np.floating[Any]]] = None,
) -> pltf.Figure:
    """
    Graph 2: Sample efficiency curves.

    Shows how data scarcity affects each method (loss at step p vs K).

    Args:
        k_values: X-axis values (support set sizes)
        maml_losses: Loss at step p for MAML at each K
        baseline_losses: Loss at step p for baseline at each K
        noise_level: Noise level used (for title/labeling)
        fixed_step: Step p at which loss was measured
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        maml_std: Standard deviation for MAML (aggregated mode)
        baseline_std: Standard deviation for baseline (aggregated mode)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogy(
        k_values, maml_losses, "b-o", label="MAML (θ*)", linewidth=2, markersize=8
    )
    ax.semilogy(
        k_values,
        baseline_losses,
        "r-s",
        label="Baseline (θ₀)",
        linewidth=2,
        markersize=8,
    )

    if maml_std is not None:
        ax.fill_between(
            k_values,
            maml_losses - maml_std,
            maml_losses + maml_std,
            color="blue",
            alpha=0.2,
        )
    if baseline_std is not None:
        ax.fill_between(
            k_values,
            baseline_losses - baseline_std,
            baseline_losses + baseline_std,
            color="red",
            alpha=0.2,
        )

    ax.set_xlabel("K (support set size)")
    ax.set_ylabel(f"Loss @ step {fixed_step}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale x-axis often useful for K values
    ax.set_xscale("log")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# =============================================================================
# Jacobian / Coefficient Recovery Graphs (Graph 7-10)
# =============================================================================


def _overlay_pred_errors(
    ax,
    coeff_values: np.ndarray,
    pred_errors: np.ndarray,
    bin_edges: np.ndarray,
    color: str,
    label: str,
):
    """Add a twinx line showing mean |prediction error| per Jacobian bin."""
    n_bins = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.clip(np.digitize(coeff_values, bin_edges) - 1, 0, n_bins - 1)

    mean_errors = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_errors[i] = pred_errors[mask].mean()

    valid = ~np.isnan(mean_errors)
    ax.plot(
        bin_centers[valid],
        mean_errors[valid],
        "-o",
        color=color,
        markersize=2,
        linewidth=1.2,
        alpha=0.8,
        label=label,
    )


# Shared color palette for estimate overlays. Both MAML and Baseline panels
# use the same colors — panels already distinguish the method.
# Each tuple is (fill_color, edge/dark_color).
_ESTIMATE_COLORS = [
    ("blue", "darkblue"),
    ("cyan", "darkcyan"),
    ("mediumpurple", "rebeccapurple"),
    ("deepskyblue", "dodgerblue"),
]


def _draw_histogram_panel(
    ax: Any,
    estimates: list[NDArray[np.floating[Any]]],
    estimate_labels: list[str],
    coeff_true: float,
    panel_title: str,
    ratio_mode: bool,
    symbol: str,
    pred_errors: Optional[list[Optional[NDArray[np.floating[Any]]]]] = None,
) -> None:
    """Draw one histogram panel (MAML or Baseline) with N overlaid estimates."""
    # Shared bins across all estimates
    all_vals = np.concatenate(estimates)
    bins = np.linspace(np.min(all_vals), np.max(all_vals), 51)

    # KDE curves + per-estimate mean lines
    x_grid = np.linspace(np.min(all_vals), np.max(all_vals), 300)
    for i, (est, label) in enumerate(zip(estimates, estimate_labels)):
        fill, edge = _ESTIMATE_COLORS[i % len(_ESTIMATE_COLORS)]
        mean, std = float(np.mean(est)), float(np.std(est))
        kde = stats.gaussian_kde(est)
        density = kde(x_grid)
        ax.fill_between(
            x_grid, density, alpha=0.4, color=fill,
            label=f"{label}: μ={mean:.4f}, σ={std:.4f}",
        )
        ax.plot(x_grid, density, color=edge, linewidth=1.2)
        ax.axvline(mean, color=fill, linestyle="-", linewidth=1.5, alpha=0.8)

    # Truth line
    truth_label = (
        "Perfect recovery (1.0)" if ratio_mode else f"True {symbol} = {coeff_true:.4f}"
    )
    ax.axvline(coeff_true, color="red", linestyle="--", linewidth=2, label=truth_label)

    # Combined mean (average of per-estimate means)
    overall_mean = float(np.mean([np.mean(e) for e in estimates]))
    ax.axvline(
        overall_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
        label=f"combined: μ={overall_mean:.4f}",
    )

    ax.set_xlabel("Recovered / True" if ratio_mode else f"{symbol} Jacobian entries")
    ax.set_ylabel("Density")
    ax.set_title(panel_title)

    # Clip x-axis to data range so distant truth lines don't squash the histogram
    data_range = float(np.max(all_vals) - np.min(all_vals))
    pad = max(data_range * 0.1, 1e-6)
    xlim_lo, xlim_hi = float(np.min(all_vals)) - pad, float(np.max(all_vals)) + pad
    ax.set_xlim(xlim_lo, xlim_hi)
    if coeff_true > xlim_hi:
        legend_loc = "upper left"
        ax.annotate(
            f"True={coeff_true:.4f} →",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            color="red",
            ha="right",
            weight="bold",
        )
    elif coeff_true < xlim_lo:
        legend_loc = "upper right"
        ax.annotate(
            f"← True={coeff_true:.4f}",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            color="red",
            ha="left",
            weight="bold",
        )
    else:
        legend_loc = "upper right"

    # Prediction error overlay
    has_pe = pred_errors is not None and any(pe is not None for pe in pred_errors)
    if has_pe:
        assert pred_errors is not None
        ax2 = ax.twinx()
        for i, (est, pe, label) in enumerate(
            zip(estimates, pred_errors, estimate_labels)
        ):
            if pe is not None:
                _, dark = _ESTIMATE_COLORS[i % len(_ESTIMATE_COLORS)]
                _overlay_pred_errors(
                    ax2, est, pe, bins, color=dark, label=f"|err| {label}"
                )
        ax2.set_ylabel("Mean |pred error|", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)
        # Merge legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc=legend_loc)
    else:
        ax.legend(fontsize=8, loc=legend_loc)


def plot_jacobian_histogram(
    maml_estimates: list[NDArray[np.floating[Any]]],
    baseline_estimates: list[NDArray[np.floating[Any]]],
    estimate_labels: list[str],
    coeff_true: float,
    title: str,
    coeff_name: str = "",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150,
    ratio_mode: bool = False,
    maml_pred_errors: Optional[list[Optional[NDArray[np.floating[Any]]]]] = None,
    baseline_pred_errors: Optional[list[Optional[NDArray[np.floating[Any]]]]] = None,
) -> pltf.Figure:
    """
    Graph 7: Jacobian Distribution Histogram with prediction error overlay.

    Side-by-side histograms (MAML left, Baseline right) showing the distribution
    of learned coefficients compared to true value. Supports N overlaid estimate
    distributions per panel (e.g. nu_u + nu_v for Navier-Stokes).

    When pred_errors are provided, a twinx axis shows mean |prediction error|
    per Jacobian bin — colored to match their corresponding histogram.

    Args:
        maml_estimates: List of N MAML coefficient arrays, one per estimate
        baseline_estimates: List of N Baseline coefficient arrays (parallel)
        estimate_labels: List of N labels for each estimate (e.g. ["nu_u", "nu_v"])
        coeff_true: True coefficient value (shared by all estimates in group)
        title: Plot title
        coeff_name: Physical coefficient name for axis labels (e.g. "nu", "D_u")
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        ratio_mode: If True, x-axis shows recovered/true ratio (aggregated mode)
        maml_pred_errors: Optional list of N per-point |pred error| arrays
        baseline_pred_errors: Optional list of N per-point |pred error| arrays

    Returns:
        matplotlib Figure object
    """
    symbol = {"nu": "ν", "D_u": "D_u", "D_v": "D_v"}.get(coeff_name, coeff_name)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if ratio_mode:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle(f"{title}\nTrue {symbol} = {coeff_true:.6f}", fontsize=12)

    _draw_histogram_panel(
        axes[0],
        maml_estimates,
        estimate_labels,
        coeff_true,
        panel_title="MAML (θ*)",
        ratio_mode=ratio_mode,
        symbol=symbol,
        pred_errors=maml_pred_errors,
    )
    _draw_histogram_panel(
        axes[1],
        baseline_estimates,
        estimate_labels,
        coeff_true,
        panel_title="Baseline (θ₀)",
        ratio_mode=ratio_mode,
        symbol=symbol,
        pred_errors=baseline_pred_errors,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_coefficient_heatmap(
    k_values: NDArray[np.integer[Any]],
    noise_levels: NDArray[np.floating[Any]],
    maml_errors: NDArray[np.floating[Any]],
    baseline_errors: NDArray[np.floating[Any]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150,
    maml_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_std: Optional[NDArray[np.floating[Any]]] = None,
) -> pltf.Figure:
    """
    Graph 8: Coefficient Error Heatmap.

    K x Noise grid showing coefficient recovery error (%) for both methods.

    Args:
        k_values: List of K values
        noise_levels: List of noise levels
        maml_errors: MAML coefficient errors (%), shape (len(noise), len(k))
        baseline_errors: Baseline coefficient errors (%), same shape
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        maml_std: Standard deviation of MAML errors (aggregated mode)
        baseline_std: Standard deviation of baseline errors (aggregated mode)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=12)

    # Format noise levels as percentages
    noise_labels = [f"{n * 100:.0f}%" for n in noise_levels]
    k_labels = [str(k) for k in k_values]

    for ax, errors, std, method, cmap in [
        (axes[0], maml_errors, maml_std, "MAML (θ*)", "Blues"),
        (axes[1], baseline_errors, baseline_std, "Baseline (θ₀)", "Oranges"),
    ]:
        im = ax.imshow(errors, cmap=cmap, aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Coefficient Error (%)")

        # Set ticks
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_labels)
        ax.set_yticks(range(len(noise_levels)))
        ax.set_yticklabels(noise_labels)

        ax.set_xlabel("K (support set size)")
        ax.set_ylabel("Noise level")
        ax.set_title(method)

        # Add text annotations
        for i in range(len(noise_levels)):
            for j in range(len(k_values)):
                val = errors[i, j]
                if std is not None:
                    text = f"{val:.1f}%\n±{std[i, j]:.1f}"
                else:
                    text = f"{val:.1f}%"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_coefficient_vs_k(
    k_values: NDArray[np.integer[Any]],
    maml_errors: NDArray[np.floating[Any]],
    baseline_errors: NDArray[np.floating[Any]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_std: Optional[NDArray[np.floating[Any]]] = None,
) -> pltf.Figure:
    """
    Graph 9: Coefficient Recovery vs K.

    Shows how coefficient recovery error changes with support set size.

    Args:
        k_values: List of K values
        maml_errors: MAML coefficient errors (%) for each K
        baseline_errors: Baseline coefficient errors (%) for each K
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        maml_std: Standard deviation for MAML (aggregated mode)
        baseline_std: Standard deviation for baseline (aggregated mode)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot MAML
    ax.plot(k_values, maml_errors, "b-o", label="MAML (θ*)", linewidth=2, markersize=8)
    if maml_std is not None:
        ax.fill_between(
            k_values,
            maml_errors - maml_std,
            maml_errors + maml_std,
            color="blue",
            alpha=0.2,
        )

    # Plot baseline
    ax.plot(
        k_values,
        baseline_errors,
        "r-s",
        label="Baseline (θ₀)",
        linewidth=2,
        markersize=8,
    )
    if baseline_std is not None:
        ax.fill_between(
            k_values,
            baseline_errors - baseline_std,
            baseline_errors + baseline_std,
            color="red",
            alpha=0.2,
        )

    ax.set_xlabel("K (support set size)")
    ax.set_ylabel("Coefficient Error (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_coefficient_vs_noise(
    noise_levels: NDArray[np.floating[Any]],
    maml_errors: NDArray[np.floating[Any]],
    baseline_errors: NDArray[np.floating[Any]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_std: Optional[NDArray[np.floating[Any]]] = None,
) -> pltf.Figure:
    """
    Graph 10: Coefficient Recovery vs Noise.

    Shows how coefficient recovery error changes with noise level.

    Args:
        noise_levels: List of noise levels (0.0, 0.01, 0.05, 0.10)
        maml_errors: MAML coefficient errors (%) for each noise level
        baseline_errors: Baseline coefficient errors (%) for each noise level
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        maml_std: Standard deviation for MAML (aggregated mode)
        baseline_std: Standard deviation for baseline (aggregated mode)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert noise to percentage for x-axis
    noise_pct = [n * 100 for n in noise_levels]

    # Plot MAML
    ax.plot(noise_pct, maml_errors, "b-o", label="MAML (θ*)", linewidth=2, markersize=8)
    if maml_std is not None:
        ax.fill_between(
            noise_pct,
            maml_errors - maml_std,
            maml_errors + maml_std,
            color="blue",
            alpha=0.2,
        )

    # Plot baseline
    ax.plot(
        noise_pct,
        baseline_errors,
        "r-s",
        label="Baseline (θ₀)",
        linewidth=2,
        markersize=8,
    )
    if baseline_std is not None:
        ax.fill_between(
            noise_pct,
            baseline_errors - baseline_std,
            baseline_errors + baseline_std,
            color="red",
            alpha=0.2,
        )

    ax.set_xlabel("Noise Level (%)")
    ax.set_ylabel("Coefficient Error (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# ─── Graph 12: Best-combo prediction scatter ──────────────────────────────────


def plot_best_combo_scatter(
    predictions: NDArray[np.floating[Any]],
    true_targets: NDArray[np.floating[Any]],
    x_pts: NDArray[np.floating[Any]],
    y_pts: NDArray[np.floating[Any]],
    steps: NDArray[np.integer[Any]],
    coeff_errors: NDArray[np.floating[Any]],
    output_index: int,
    output_label: str,
    title: str,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> pltf.Figure:
    """
    Prediction evolution during fine-tuning using tricontourf.

    Layout:
      Row 0: True reference panel (tricontourf) — shown once.
      Rows 1..N: Predicted (tricontourf) + Pred-vs-True scatter.

    Points are multi-snapshot holdout samples (not a single timestep).
    Colorscale is percentile-clipped and symmetric around 0.
    """
    n_steps = len(steps)
    true_col = true_targets[:, output_index]

    # Shared triangulation + percentile-clipped symmetric colorscale
    triang = mtri.Triangulation(x_pts, y_pts)
    plo, phi = float(np.percentile(true_col, 2)), float(np.percentile(true_col, 98))
    vlim = max(abs(plo), abs(phi))
    if vlim == 0:
        vlim = 1.0
    levels = np.linspace(-vlim, vlim, 64)

    # Row 0: true reference.  Rows 1..N: predicted + correlation scatter.
    total_rows = n_steps + 1
    fig, axes = plt.subplots(total_rows, 2, figsize=(10, 3.0 * total_rows),
                             squeeze=False)

    # ── Row 0: True reference (col 0), col 1 off ──
    ax_true = axes[0, 0]
    ax_true.tricontourf(triang, true_col, levels=levels, cmap="RdBu_r",
                        extend="both")
    ax_true.set_aspect("equal")
    ax_true.set_title(f"True {output_label}  (multi-snapshot holdout)", fontsize=10)
    ax_true.set_ylabel("Reference", fontsize=9, fontweight="bold")
    axes[0, 1].axis("off")

    # ── Per-step rows ──
    for i, step in enumerate(steps):
        pred_col = predictions[i, :, output_index]
        row = i + 1

        # Col 0: Predicted tricontourf (same levels as true)
        ax_pred = axes[row, 0]
        ax_pred.tricontourf(triang, pred_col, levels=levels, cmap="RdBu_r",
                            extend="both")
        ax_pred.set_aspect("equal")
        if i == 0:
            ax_pred.set_title(f"Predicted {output_label}", fontsize=10)

        # Row label
        step_int = int(step)
        err = coeff_errors[i]
        err_str = f"{err:.1f}%" if not np.isnan(err) else "n/a"
        ax_pred.set_ylabel(f"Step {step_int}\nerr={err_str}",
                           fontsize=9, fontweight="bold")

        # Col 1: Correlation scatter
        ax_corr = axes[row, 1]
        ax_corr.scatter(true_col, pred_col, s=1, alpha=0.3, rasterized=True)
        lo = min(float(true_col.min()), float(pred_col.min()))
        hi = max(float(true_col.max()), float(pred_col.max()))
        margin = (hi - lo) * 0.05
        ref = [lo - margin, hi + margin]
        ax_corr.plot(ref, ref, "k--", alpha=0.3, linewidth=1)
        ax_corr.set_xlim(ref)
        ax_corr.set_ylim(ref)
        ax_corr.set_aspect("equal")
        if i == 0:
            ax_corr.set_title("Pred vs True", fontsize=10)
        ax_corr.set_xlabel(f"True {output_label}", fontsize=8)
        ax_corr.set_ylabel(f"Pred {output_label}", fontsize=8)

        # R²
        ss_res = float(np.sum((pred_col - true_col) ** 2))
        ss_tot = float(np.sum((true_col - np.mean(true_col)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        ax_corr.annotate(f"R²={r2:.4f}", xy=(0.05, 0.92), xycoords="axes fraction",
                         fontsize=8, fontweight="bold")

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# ─── Graph 11: Cross-experiment coefficient scatter grid ──────────────────────


# Type alias for one panel's model data
ModelScatterData = list[
    Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], list[str], str]
]
# Each element: (true_vals, recovered_vals, task_names, label)

PanelDataDict = dict[Tuple[str, float, int], ModelScatterData]
# Key: (coeff_name, noise_level, k_value) → list of model data


def _ic_type(task_name: str) -> str:
    """Extract IC type code from task name like 'br2t_gp_001_fourier'."""
    parts = task_name.replace("_fourier", "").split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def _scatter_panel(
    ax: pltax.Axes,
    model_data: ModelScatterData,
    coeff_name: str,
) -> None:
    """Render one scatter panel with overlaid models, regression lines, Pearson r."""
    # Build IC color map dynamically from all task names in this panel
    all_ic_types: set[str] = set()
    for _, _, task_names, _ in model_data:
        all_ic_types.update(_ic_type(n) for n in task_names)
    ic_color_map = {
        ic: IC_PALETTE[j % len(IC_PALETTE)] for j, ic in enumerate(sorted(all_ic_types))
    }

    all_true: list[float] = []
    all_rec: list[float] = []

    for i, (true_vals, recovered_vals, task_names, label) in enumerate(model_data):
        if len(true_vals) == 0:
            continue

        all_true.extend(true_vals.tolist())
        all_rec.extend(recovered_vals.tolist())

        # Each model entry (MAML and BL separately) gets its own marker
        exp_idx = i
        is_baseline = "(BL)" in label
        ic_types = [_ic_type(n) for n in task_names]

        # Scatter: border colored by IC type, no fill.
        # Each experiment gets its own marker from MODEL_MARKERS.
        marker_style = MODEL_MARKERS[exp_idx % len(MODEL_MARKERS)]
        marker_size = 120

        for ic, color in ic_color_map.items():
            mask = np.array([t == ic for t in ic_types])
            if not mask.any():
                continue
            ax.scatter(
                true_vals[mask],
                recovered_vals[mask],
                facecolors="none",
                edgecolors=color,
                marker=marker_style,
                s=marker_size,
                alpha=0.85,
                linewidths=1.5 if not is_baseline else 0.8,
            )

        # Label each point centered inside the marker
        for j, tname in enumerate(task_names):
            short = tname.split("_fourier")[0].split("heat_")[-1]
            parts = short.split("_")
            short_label = parts[-1] if len(parts) >= 2 else short
            ax.text(
                true_vals[j], recovered_vals[j], short_label,
                fontsize=4, alpha=0.7,
                ha="center", va="center",
            )

        # Regression line + Pearson r (colored per experiment, solid=MAML, dashed=baseline)
        if len(true_vals) >= 2:
            r, _ = stats.pearsonr(true_vals, recovered_vals)
            slope, intercept = np.polyfit(true_vals, recovered_vals, 1)
            x_fit = np.linspace(true_vals.min(), true_vals.max(), 100)
            linestyle = "--" if is_baseline else "-"
            color_list = MODEL_COLORS_LIGHT if is_baseline else MODEL_COLORS_DARK
            line_color = color_list[exp_idx % len(color_list)]
            ax.plot(
                x_fit,
                (slope * x_fit) + intercept,
                color=line_color,
                linewidth=2,
                alpha=0.7,
                linestyle=linestyle,
                label=f"{label}: r={r:.2f}, slope={slope:.2f}",
            )

    if not all_true:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # y=x reference line
    all_vals = np.array(all_true + all_rec)
    lo, hi = float(all_vals.min()), float(all_vals.max())
    margin = (hi - lo) * 0.1
    ref = [lo - margin, hi + margin]
    ax.plot(ref, ref, "k--", alpha=0.25, linewidth=1)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel(f"True {coeff_name}", fontsize=9)


def plot_coefficient_scatter_grid(
    panel_data: PanelDataDict,
    coeff_names: list[str],
    k_values: list[int],
    noise_levels: list[float],
    save_path: Optional[Path] = None,
    dpi: int = 150,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.5),
    step: Optional[int] = None,
) -> pltf.Figure:
    """
    Multi-panel scatter grid: true vs recovered coefficients across experiments.

    Rows = coefficient_names × noise_levels, Columns = k_values.
    Each panel shows one scatter per model, colored by IC type.
    """
    n_rows = len(coeff_names) * len(noise_levels)
    n_cols = len(k_values)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    row_idx = 0
    for coeff_name in coeff_names:
        for noise in noise_levels:
            for col_idx, k_val in enumerate(k_values):
                ax = axes[row_idx, col_idx]
                key = (coeff_name, noise, k_val)
                models = panel_data.get(key, [])

                _scatter_panel(ax, models, coeff_name)

                # Column headers on top row
                if row_idx == 0:
                    ax.set_title(f"K = {k_val}", fontsize=11)

                # Row labels on leftmost column
                if col_idx == 0:
                    ax.set_ylabel(
                        f"Recovered {coeff_name}",
                        fontsize=10,
                        fontweight="bold",
                    )
                    ax.annotate(
                        f"noise={noise:.0%}",
                        xy=(0, 0.5),
                        xycoords="axes fraction",
                        xytext=(-65, 0),
                        textcoords="offset points",
                        fontsize=9,
                        fontstyle="italic",
                        ha="right",
                        va="center",
                        rotation=90,
                    )

            row_idx += 1

    step_str = f" — Step {step}" if step is not None else ""
    fig.suptitle(
        f"Coefficient Recovery: True vs Recovered{step_str}",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()

    # --- Top legend: IC type colors (neutral circle marker) ---
    # Collect all IC types from first panel's data
    first_key = next(iter(panel_data), None)
    all_ic_types: set[str] = set()
    if first_key is not None:
        for _, _, task_names, _ in panel_data[first_key]:
            all_ic_types.update(_ic_type(n) for n in task_names)
    ic_color_map = {
        ic: IC_PALETTE[j % len(IC_PALETTE)]
        for j, ic in enumerate(sorted(all_ic_types))
    }
    if ic_color_map:
        ic_handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color,
                markeredgecolor="k", markeredgewidth=0.4, markersize=8, label=ic,
            )
            for ic, color in ic_color_map.items()
        ]
        fig.legend(
            handles=ic_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(ic_handles),
            fontsize=8,
            framealpha=0.9,
            title="IC type",
            title_fontsize=9,
        )

    # --- Bottom legend: model markers + regression line info ---
    # Harvest regression-line handles from all panels (different K columns have different models)
    reg_handles = []
    seen_model_names: set[str] = set()
    for row in axes:
        for ax in row:
            for handle, lbl in zip(*ax.get_legend_handles_labels()):
                # Dedup by model name (before the ": r=..." stats)
                model_name = lbl.split(":")[0].strip() if ":" in lbl else lbl
                if model_name not in seen_model_names:
                    reg_handles.append(handle)
                    seen_model_names.add(model_name)
    if reg_handles and first_key is not None:
        # Build marker-shape handles for each model entry (MAML + BL each get own marker)
        all_models = []
        for key in panel_data:
            for entry in panel_data[key]:
                if entry[3] not in {e[3] for e in all_models}:
                    all_models.append(entry)
        marker_handles = []
        for i, (_, _, _, label) in enumerate(all_models):
            marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
            # Use short label (strip stats)
            short = label.split(":")[0].strip() if ":" in label else label
            marker_handles.append(
                Line2D(
                    [0], [0], marker=marker, color="w", markerfacecolor="none",
                    markeredgecolor="grey", markeredgewidth=1.2, markersize=8,
                    label=short,
                )
            )

        all_bottom = marker_handles + reg_handles
        all_bottom_labels = [h.get_label() for h in all_bottom]
        fig.legend(
            handles=all_bottom,
            labels=all_bottom_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(all_bottom), 4),
            fontsize=7,
            framealpha=0.9,
            title="Models",
            title_fontsize=9,
        )

    # Draw horizontal separators between coefficient groups
    n_noise = len(noise_levels)
    if len(coeff_names) > 1:
        for group_idx in range(1, len(coeff_names)):
            # Place line midway between last row of prev group and first row of next
            prev_last_row = (group_idx * n_noise) - 1
            next_first_row = group_idx * n_noise
            y_top = axes[prev_last_row, 0].get_position().y0
            y_bot = axes[next_first_row, 0].get_position().y1
            y_sep = (y_top + y_bot) / 2
            line = Line2D(
                [0.0, 1.0],
                [y_sep, y_sep],
                transform=fig.transFigure,
                color="black",
                linewidth=2.5,
                linestyle="-",
                alpha=0.7,
            )
            fig.add_artist(line)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
