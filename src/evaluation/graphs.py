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
from scipy import stats

from src.evaluation.metrics import steps_to_plateau

# Cross-experiment scatter plot constants
IC_COLORS: dict[str, str] = {
    "gp": "#1f77b4",
    "pu": "#ff7f0e",
    "rs": "#2ca02c",
    "lp": "#d62728",
    "mp": "#9467bd",
}
MODEL_MARKERS: list[str] = ["o", "^", "*", "D", "v", "s", "P", "X"]
MODEL_COLORS: list[str] = [
    "#e74c3c", "#3498db", "#9ab042", "#e67e22",
    "#8e44ad", "#1abc9c", "#f39c12", "#2c3e50",
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
                ax.axvline(x=s - 1, color=color, linewidth=1.0, alpha=alpha)

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
        maml_plateau_step, maml_holdout[maml_plateau_step],
        "bo", markersize=8, label=f"Plateau @ {maml_plateau_step}", zorder=5,
    )
    ax1.axvline(x=maml_plateau_step, color="blue", linestyle=":", alpha=0.5)

    ax2.plot(
        baseline_plateau_step, baseline_holdout[baseline_plateau_step],
        "rs", markersize=8, label=f"Plateau @ {baseline_plateau_step}", zorder=5,
    )
    ax2.axvline(x=baseline_plateau_step, color="red", linestyle=":", alpha=0.5)

    # Plateau std bands (aggregated mode)
    if plateau_step_std is not None:
        maml_step_std, baseline_step_std = plateau_step_std
        ax1.axvspan(
            maml_plateau_step - maml_step_std,
            maml_plateau_step + maml_step_std,
            color="blue", alpha=0.1,
        )
        ax2.axvspan(
            baseline_plateau_step - baseline_step_std,
            baseline_plateau_step + baseline_step_std,
            color="red", alpha=0.1,
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
        color=min_line_color, linewidth=min_linewidth, alpha=min_line_alpha,
    )
    ax2.axhline(
        y=min_holdout_loss,
        color=min_line_color, linewidth=min_linewidth, alpha=min_line_alpha,
    )
    ax2.annotate(
        f"min: {min_holdout_loss:.2e}",
        xy=(steps[-1], min_holdout_loss),
        fontsize=7, color=min_line_color, alpha=min_line_alpha,
        va="bottom", ha="right",
    )

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_speedup_heatmap(
    speedups: NDArray[np.floating[Any]],
    k_values: NDArray[np.integer[Any]],
    noise_levels: NDArray[np.floating[Any]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    std_values: Optional[NDArray[np.floating[Any]]] = None,
    maml_losses: Optional[NDArray[np.floating[Any]]] = None,
    baseline_losses: Optional[NDArray[np.floating[Any]]] = None,
    inf_counts: Optional[NDArray[np.floating[Any]]] = None,
    n_total: Optional[int] = None,
) -> pltf.Figure:
    """
    Graph 3: Speedup heatmap (K × Noise).

    Cell value = baseline_steps_to_plateau / maml_steps_to_plateau.
    Values > 1 mean MAML reaches its plateau faster.

    Args:
        speedups: 2D array of speedup ratios, shape (n_noise, n_K)
        k_values: K values for x-axis labels
        noise_levels: Noise levels for y-axis labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        std_values: Optional std array for annotations (aggregated mode)
        maml_losses: Optional 2D array of MAML's loss at plateau per cell
        baseline_losses: Optional 2D array of baseline's loss at plateau per cell
        inf_counts: Optional 2D array of inf counts per cell (aggregated mode)
        n_total: Total number of tasks aggregated (for showing inf ratio)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle NaN/None values for display
    mask = np.isnan(speedups)

    # Create heatmap
    im = ax.imshow(speedups, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup Ratio (>1 = MAML faster)")

    # Axis labels
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(noise_levels)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([f"{n:.0%}" for n in noise_levels])
    ax.set_xlabel("K (support set size)")
    ax.set_ylabel("Noise Level")

    # Check if we have loss values to display
    show_losses = maml_losses is not None and baseline_losses is not None
    has_inf_counts = inf_counts is not None

    # Annotate cells with values
    for i in range(len(noise_levels)):
        for j in range(len(k_values)):
            if mask[i, j]:
                text = "N/A"
                color = "gray"
            else:
                value = speedups[i, j]
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
                        text = f"∞ ({n_inf}/{n_total})\nμ={value:.2f}x ±{std_values[i, j]:.2f}"
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
                    text = f"{value:.2f}x\n±{std_values[i, j]:.2f}"
                    color = "black" if 0.8 < value < 2.0 else "white"
                elif show_losses:
                    assert maml_losses is not None and baseline_losses is not None
                    m_loss = maml_losses[i, j]
                    b_loss = baseline_losses[i, j]
                    text = f"{value:.2f}x\n({m_loss:.1e} / {b_loss:.1e})"
                    color = "black" if 0.8 < value < 2.0 else "white"
                else:
                    text = f"{value:.2f}x"
                    color = "black" if 0.8 < value < 2.0 else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title)
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
    bins_list: list[float] = bins.tolist()

    # Overlaid histograms + per-estimate mean lines
    for i, (est, label) in enumerate(zip(estimates, estimate_labels)):
        fill, edge = _ESTIMATE_COLORS[i % len(_ESTIMATE_COLORS)]
        mean, std = float(np.mean(est)), float(np.std(est))
        ax.hist(
            est,
            bins=bins_list,
            alpha=0.6,
            color=fill,
            edgecolor=edge,
            label=f"{label}: μ={mean:.4f}, σ={std:.4f}",
        )
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
    ax.set_ylabel("Count")
    ax.set_title(panel_title)

    # Clip x-axis to data range so distant truth lines don't squash the histogram
    data_range = float(np.max(all_vals) - np.min(all_vals))
    pad = max(data_range * 0.1, 1e-6)
    xlim_lo, xlim_hi = float(np.min(all_vals)) - pad, float(np.max(all_vals)) + pad
    ax.set_xlim(xlim_lo, xlim_hi)
    if coeff_true > xlim_hi:
        legend_loc = "upper left"
        ax.annotate(
            f"True={coeff_true:.4f} →", xy=(0.98, 0.95),
            xycoords="axes fraction", fontsize=8, color="red",
            ha="right", weight="bold",
        )
    elif coeff_true < xlim_lo:
        legend_loc = "upper right"
        ax.annotate(
            f"← True={coeff_true:.4f}", xy=(0.02, 0.95),
            xycoords="axes fraction", fontsize=8, color="red",
            ha="left", weight="bold",
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
    show_ic_legend: bool,
) -> None:
    """Render one scatter panel with overlaid models, regression lines, Pearson r."""
    all_true: list[float] = []
    all_rec: list[float] = []

    for i, (true_vals, recovered_vals, task_names, label) in enumerate(model_data):
        if len(true_vals) == 0:
            continue

        all_true.extend(true_vals.tolist())
        all_rec.extend(recovered_vals.tolist())

        marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        ic_types = [_ic_type(n) for n in task_names]

        # Scatter by IC type
        for ic, color in IC_COLORS.items():
            mask = np.array([t == ic for t in ic_types])
            if not mask.any():
                continue
            ic_label = ic if (show_ic_legend and i == 0) else None
            ax.scatter(
                true_vals[mask], recovered_vals[mask],
                c=color, marker=marker, label=ic_label,
                s=45, alpha=0.75, edgecolors="k", linewidths=0.4,
            )

        # Regression line + Pearson r
        if len(true_vals) >= 2:
            r, _ = stats.pearsonr(true_vals, recovered_vals)
            slope, intercept = np.polyfit(true_vals, recovered_vals, 1)
            x_fit = np.linspace(true_vals.min(), true_vals.max(), 100)
            line_color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.plot(
                x_fit, (slope * x_fit) + intercept,
                color=line_color, linewidth=2, alpha=0.7,
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
        n_rows, n_cols,
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

                # Show IC legend only on first panel (for fig.legend harvesting)
                show_ic_legend = (row_idx == 0 and col_idx == 0)
                _scatter_panel(ax, models, coeff_name, show_ic_legend)

                # Column headers on top row
                if row_idx == 0:
                    ax.set_title(f"K = {k_val}", fontsize=11)

                # Row labels on leftmost column
                if col_idx == 0:
                    ax.set_ylabel(
                        f"Recovered {coeff_name}",
                        fontsize=10, fontweight="bold",
                    )
                    ax.annotate(
                        f"noise={noise:.0%}",
                        xy=(0, 0.5), xycoords="axes fraction",
                        xytext=(-65, 0), textcoords="offset points",
                        fontsize=9, fontstyle="italic",
                        ha="right", va="center", rotation=90,
                    )

            row_idx += 1

    step_str = f" — Step {step}" if step is not None else ""
    fig.suptitle(
        f"Coefficient Recovery: True vs Recovered{step_str}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    # Figure-level legend harvested from first panel (IC colors + model lines)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles), 6),
            fontsize=8, framealpha=0.9,
        )

    # Draw horizontal separators between coefficient groups
    n_noise = len(noise_levels)
    if len(coeff_names) > 1:
        for group_idx in range(1, len(coeff_names)):
            # Place line just above the first row of the next coefficient group
            bot_row = group_idx * n_noise
            y_sep = axes[bot_row, 0].get_position().y1 - 0.002
            line = Line2D(
                [0.0, 1.0], [y_sep, y_sep],
                transform=fig.transFigure,
                color="black", linewidth=2.5, linestyle="-", alpha=0.7,
            )
            fig.add_artist(line)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
