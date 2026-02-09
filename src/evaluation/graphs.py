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
from matplotlib.patches import Rectangle

from src.evaluation.metrics import steps_to_plateau


def plot_convergence(
    maml_losses: NDArray[np.floating[Any]],
    baseline_losses: NDArray[np.floating[Any]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[NDArray[np.floating[Any]]] = None,
    baseline_std: Optional[NDArray[np.floating[Any]]] = None,
    maml_min_step: Optional[int] = None,
    baseline_min_step: Optional[int] = None,
    min_step_std: Optional[Tuple[float, float]] = None,
    deriv_threshold: float = 1e-7,
    k_shot: Optional[int] = None,
) -> pltf.Figure:
    """
    Graph 5: Convergence plot (Loss vs Steps).

    Shows full training trajectory for both methods with markers at minimum.

    Args:
        maml_losses: MAML loss curve (or mean if aggregated)
        baseline_losses: Baseline loss curve (or mean if aggregated)
        title: Plot title
        save_path: Path to save figure (None = don't save)
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        maml_std: Standard deviation for MAML losses (aggregated mode)
        baseline_std: Standard deviation for baseline losses (aggregated mode)
        maml_min_step: Step where MAML hits minimum (for marker)
        baseline_min_step: Step where baseline hits minimum (for marker)
        min_step_std: Tuple of (maml_step_std, baseline_step_std) for vertical bands
        k_shot: Number of support samples (for y-axis label)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = np.arange(len(maml_losses))

    # Plot curves
    ax.semilogy(steps, maml_losses, "b-", label="MAML (θ*)", linewidth=2)
    ax.semilogy(steps, baseline_losses, "r-", label="Baseline (θ₀)", linewidth=2)

    # Add std bands if provided (aggregated mode)
    if maml_std is not None:
        ax.fill_between(
            steps,
            maml_losses - maml_std,
            maml_losses + maml_std,
            color="blue",
            alpha=0.2,
        )
    if baseline_std is not None:
        ax.fill_between(
            steps,
            baseline_losses - baseline_std,
            baseline_losses + baseline_std,
            color="red",
            alpha=0.2,
        )

    # Mark plateau points
    if maml_min_step is None:
        maml_min_step = steps_to_plateau(maml_losses, deriv_threshold=deriv_threshold)
    if baseline_min_step is None:
        baseline_min_step = steps_to_plateau(
            baseline_losses, deriv_threshold=deriv_threshold
        )

    # Plot markers at minimum
    ax.plot(
        maml_min_step,
        maml_losses[maml_min_step],
        "bo",
        markersize=10,
        label=f"MAML min @ {maml_min_step}",
        zorder=5,
    )
    ax.plot(
        baseline_min_step,
        baseline_losses[baseline_min_step],
        "rs",
        markersize=10,
        label=f"Baseline min @ {baseline_min_step}",
        zorder=5,
    )

    # Add vertical lines at minimum steps
    ax.axvline(x=maml_min_step, color="blue", linestyle=":", alpha=0.5)
    ax.axvline(x=baseline_min_step, color="red", linestyle=":", alpha=0.5)

    # Add std bands for step positions (aggregated mode)
    if min_step_std is not None:
        maml_step_std, baseline_step_std = min_step_std
        ax.axvspan(
            maml_min_step - maml_step_std,
            maml_min_step + maml_step_std,
            color="blue",
            alpha=0.1,
        )
        ax.axvspan(
            baseline_min_step - baseline_step_std,
            baseline_min_step + baseline_step_std,
            color="red",
            alpha=0.1,
        )

    ax.set_xlabel("Gradient Steps")
    ylabel = f"MSE Loss (K={k_shot})" if k_shot is not None else "MSE Loss"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


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
) -> pltf.Figure:
    """
    Graph: Train vs Holdout convergence comparison.

    Shows generalization gap - if train << holdout, the model is overfitting.

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

    ax1.set_xlabel("Gradient Steps")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("MAML (θ*)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

    ax2.set_xlabel("Gradient Steps")
    ax2.set_title("Baseline (θ₀)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Draw thin horizontal line at overall minimum holdout loss (spans both subplots)
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


def plot_jacobian_histogram(
    maml_coeff_1: NDArray[np.floating[Any]],
    maml_coeff_2: NDArray[np.floating[Any]],
    baseline_coeff_1: NDArray[np.floating[Any]],
    baseline_coeff_2: NDArray[np.floating[Any]],
    coeff_true: float,
    title: str,
    coeff_name: str = "nu",
    coeff_1_label: str = "u-eq",
    coeff_2_label: str = "v-eq",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150,
    maml_pred_errors_1: Optional[NDArray[np.floating[Any]]] = None,
    maml_pred_errors_2: Optional[NDArray[np.floating[Any]]] = None,
    baseline_pred_errors_1: Optional[NDArray[np.floating[Any]]] = None,
    baseline_pred_errors_2: Optional[NDArray[np.floating[Any]]] = None,
) -> pltf.Figure:
    """
    Graph 7: Jacobian Distribution Histogram with prediction error overlay.

    Side-by-side histograms showing the distribution of learned diffusion
    coefficients compared to true value. Overlays estimates from u-equation
    and v-equation to reveal if they learned different coefficients.

    When pred_errors are provided, a twinx axis shows mean |prediction error|
    per Jacobian bin — colored to match their corresponding histogram.

    Args:
        maml_coeff_1: MAML coefficient from first equation (nu_u or D_u)
        maml_coeff_2: MAML coefficient from second equation (nu_v or D_v)
        baseline_coeff_1: Baseline coefficient from first equation
        baseline_coeff_2: Baseline coefficient from second equation
        coeff_true: True coefficient value
        title: Plot title
        coeff_name: Coefficient name ('nu', 'D_u', or 'D_v') for axis labels
        coeff_1_label: Label for first equation (default 'u-eq')
        coeff_2_label: Label for second equation (default 'v-eq')
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        maml_pred_errors_1: Per-point |pred error| for eq1 (u_t), shape (n,)
        maml_pred_errors_2: Per-point |pred error| for eq2 (v_t), shape (n,)
        baseline_pred_errors_1: Per-point |pred error| for eq1, shape (n,)
        baseline_pred_errors_2: Per-point |pred error| for eq2, shape (n,)

    Returns:
        matplotlib Figure object
    """
    # Symbol map for display
    symbol = {"nu": "ν", "D_u": "D_u", "D_v": "D_v"}.get(coeff_name, coeff_name)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{title}\nTrue {symbol} = {coeff_true:.6f}", fontsize=12)

    # Compute stats for each
    maml_1_mean, maml_1_std = np.mean(maml_coeff_1), np.std(maml_coeff_1)
    maml_2_mean, maml_2_std = np.mean(maml_coeff_2), np.std(maml_coeff_2)
    baseline_1_mean, baseline_1_std = (
        np.mean(baseline_coeff_1),
        np.std(baseline_coeff_1),
    )
    baseline_2_mean, baseline_2_std = (
        np.mean(baseline_coeff_2),
        np.std(baseline_coeff_2),
    )

    # Compute shared bin edges for fair comparison
    all_maml = np.concatenate([maml_coeff_1, maml_coeff_2])
    all_baseline = np.concatenate([baseline_coeff_1, baseline_coeff_2])
    maml_bins = np.linspace(np.min(all_maml), np.max(all_maml), 51)
    baseline_bins = np.linspace(np.min(all_baseline), np.max(all_baseline), 51)

    # MAML histogram - overlay both equations
    ax = axes[0]
    ax.hist(
        maml_coeff_1,
        bins=maml_bins,
        alpha=0.6,
        color="blue",
        edgecolor="darkblue",
        label=f"{coeff_1_label}: μ={maml_1_mean:.4f}, σ={maml_1_std:.4f}",
    )
    ax.hist(
        maml_coeff_2,
        bins=maml_bins,
        alpha=0.6,
        color="cyan",
        edgecolor="darkcyan",
        label=f"{coeff_2_label}: μ={maml_2_mean:.4f}, σ={maml_2_std:.4f}",
    )
    ax.axvline(
        coeff_true,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True {symbol} = {coeff_true:.4f}",
    )
    ax.axvline(maml_1_mean, color="blue", linestyle="-", linewidth=1.5, alpha=0.8)
    ax.axvline(maml_2_mean, color="darkcyan", linestyle="-", linewidth=1.5, alpha=0.8)
    maml_overall_mean = (maml_1_mean + maml_2_mean) / 2
    ax.axvline(
        maml_overall_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
        label=f"combined: μ={maml_overall_mean:.4f}",
    )
    ax.set_xlabel(f"{symbol} Jacobian entries")
    ax.set_ylabel("Count")
    ax.set_title("MAML (θ*)")

    # Prediction error overlay on MAML panel
    if maml_pred_errors_1 is not None:
        ax2 = ax.twinx()
        _overlay_pred_errors(
            ax2,
            maml_coeff_1,
            maml_pred_errors_1,
            maml_bins,
            color="darkblue",
            label=f"|err| {coeff_1_label}",
        )
        if maml_pred_errors_2 is not None:
            _overlay_pred_errors(
                ax2,
                maml_coeff_2,
                maml_pred_errors_2,
                maml_bins,
                color="darkcyan",
                label=f"|err| {coeff_2_label}",
            )
        ax2.set_ylabel("Mean |pred error|", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)
        # Merge legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    else:
        ax.legend(fontsize=8)

    # Baseline histogram - overlay both equations
    ax = axes[1]
    ax.hist(
        baseline_coeff_1,
        bins=baseline_bins,
        alpha=0.6,
        color="orange",
        edgecolor="darkorange",
        label=f"{coeff_1_label}: μ={baseline_1_mean:.4f}, σ={baseline_1_std:.4f}",
    )
    ax.hist(
        baseline_coeff_2,
        bins=baseline_bins,
        alpha=0.6,
        color="yellow",
        edgecolor="gold",
        label=f"{coeff_2_label}: μ={baseline_2_mean:.4f}, σ={baseline_2_std:.4f}",
    )
    ax.axvline(
        coeff_true,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True {symbol} = {coeff_true:.4f}",
    )
    ax.axvline(
        baseline_1_mean, color="darkorange", linestyle="-", linewidth=1.5, alpha=0.8
    )
    ax.axvline(baseline_2_mean, color="gold", linestyle="-", linewidth=1.5, alpha=0.8)
    baseline_overall_mean = (baseline_1_mean + baseline_2_mean) / 2
    ax.axvline(
        baseline_overall_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
        label=f"combined: μ={baseline_overall_mean:.4f}",
    )
    ax.set_xlabel(f"{symbol} Jacobian entries")
    ax.set_ylabel("Count")
    ax.set_title("Baseline (θ₀)")

    # Prediction error overlay on Baseline panel
    if baseline_pred_errors_1 is not None:
        ax2 = ax.twinx()
        _overlay_pred_errors(
            ax2,
            baseline_coeff_1,
            baseline_pred_errors_1,
            baseline_bins,
            color="darkorange",
            label=f"|err| {coeff_1_label}",
        )
        if baseline_pred_errors_2 is not None:
            _overlay_pred_errors(
                ax2,
                baseline_coeff_2,
                baseline_pred_errors_2,
                baseline_bins,
                color="goldenrod",
                label=f"|err| {coeff_2_label}",
            )
        ax2.set_ylabel("Mean |pred error|", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    else:
        ax.legend(fontsize=8)

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
