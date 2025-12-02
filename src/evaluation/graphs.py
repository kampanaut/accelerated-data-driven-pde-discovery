"""
Visualization functions for MAML experiment results.

Graph types from experiment_bible.md:
- Graph 1: Noise robustness curves (steps to threshold vs noise level)
- Graph 2: Sample efficiency curves (loss at step p vs K)
- Graph 3: Speedup heatmap (K × Noise grid)
- Graph 4: Loss ratio heatmap (K × Noise grid)
- Graph 5: Convergence plot (loss vs steps)

Each function supports both single-task and aggregated (mean ± std) modes.
"""

from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.evaluation.metrics import steps_to_plateau


def plot_convergence(
    maml_losses: List[float],
    baseline_losses: List[float],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[List[float]] = None,
    baseline_std: Optional[List[float]] = None,
    maml_min_step: Optional[int] = None,
    baseline_min_step: Optional[int] = None,
    min_step_std: Optional[Tuple[float, float]] = None,
    deriv_threshold: float = 1e-7
) -> plt.Figure:
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

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = np.arange(len(maml_losses))
    maml_losses = np.array(maml_losses)
    baseline_losses = np.array(baseline_losses)

    # Plot curves
    ax.semilogy(steps, maml_losses, 'b-', label='MAML (θ*)', linewidth=2)
    ax.semilogy(steps, baseline_losses, 'r-', label='Baseline (θ₀)', linewidth=2)

    # Add std bands if provided (aggregated mode)
    if maml_std is not None:
        maml_std = np.array(maml_std)
        ax.fill_between(
            steps,
            maml_losses - maml_std,
            maml_losses + maml_std,
            color='blue', alpha=0.2
        )
    if baseline_std is not None:
        baseline_std = np.array(baseline_std)
        ax.fill_between(
            steps,
            baseline_losses - baseline_std,
            baseline_losses + baseline_std,
            color='red', alpha=0.2
        )

    # Mark plateau points
    if maml_min_step is None:
        maml_min_step = steps_to_plateau(maml_losses, deriv_threshold=deriv_threshold)
    if baseline_min_step is None:
        baseline_min_step = steps_to_plateau(baseline_losses, deriv_threshold=deriv_threshold)

    # Plot markers at minimum
    ax.plot(maml_min_step, maml_losses[maml_min_step], 'bo', markersize=10,
            label=f'MAML min @ {maml_min_step}', zorder=5)
    ax.plot(baseline_min_step, baseline_losses[baseline_min_step], 'rs', markersize=10,
            label=f'Baseline min @ {baseline_min_step}', zorder=5)

    # Add vertical lines at minimum steps
    ax.axvline(x=maml_min_step, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=baseline_min_step, color='red', linestyle=':', alpha=0.5)

    # Add std bands for step positions (aggregated mode)
    if min_step_std is not None:
        maml_step_std, baseline_step_std = min_step_std
        ax.axvspan(maml_min_step - maml_step_std, maml_min_step + maml_step_std,
                   color='blue', alpha=0.1)
        ax.axvspan(baseline_min_step - baseline_step_std, baseline_min_step + baseline_step_std,
                   color='red', alpha=0.1)

    ax.set_xlabel('Gradient Steps')
    ax.set_ylabel('MSE Loss')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_train_holdout_convergence(
    maml_train: List[float],
    maml_holdout: List[float],
    baseline_train: List[float],
    baseline_holdout: List[float],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
) -> plt.Figure:
    """
    Graph: Train vs Holdout convergence comparison.

    Shows generalization gap - if train << holdout, the model is overfitting.

    Args:
        maml_train: MAML training loss curve
        maml_holdout: MAML holdout loss curve
        baseline_train: Baseline training loss curve
        baseline_holdout: Baseline holdout loss curve
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size in inches
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    steps = np.arange(len(maml_train))

    # MAML subplot
    ax1.semilogy(steps, maml_train, 'b-', label='Train', linewidth=2)
    ax1.semilogy(steps, maml_holdout, 'b--', label='Holdout', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Gradient Steps')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('MAML (θ*)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Baseline subplot
    ax2.semilogy(steps, baseline_train, 'r-', label='Train', linewidth=2)
    ax2.semilogy(steps, baseline_holdout, 'r--', label='Holdout', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Gradient Steps')
    ax2.set_title('Baseline (θ₀)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_speedup_heatmap(
    speedups: np.ndarray,
    k_values: List[int],
    noise_levels: List[float],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    std_values: Optional[np.ndarray] = None,
    maml_losses: Optional[np.ndarray] = None,
    baseline_losses: Optional[np.ndarray] = None,
    inf_counts: Optional[np.ndarray] = None,
    n_total: Optional[int] = None,
) -> plt.Figure:
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
    speedups = np.array(speedups, dtype=float)
    mask = np.isnan(speedups)

    # Create heatmap
    im = ax.imshow(speedups, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup Ratio (>1 = MAML faster)')

    # Axis labels
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(noise_levels)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([f'{n:.0%}' for n in noise_levels])
    ax.set_xlabel('K (support set size)')
    ax.set_ylabel('Noise Level')

    # Check if we have loss values to display
    show_losses = maml_losses is not None and baseline_losses is not None
    has_inf_counts = inf_counts is not None

    # Annotate cells with values
    for i in range(len(noise_levels)):
        for j in range(len(k_values)):
            if mask[i, j]:
                text = 'N/A'
                color = 'gray'
            else:
                value = speedups[i, j]
                n_inf = int(inf_counts[i, j]) if has_inf_counts else 0

                # Aggregated mode with some inf values
                if has_inf_counts and n_inf > 0:
                    # Draw turquoise background
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      facecolor='#00CED1', edgecolor='none')
                    ax.add_patch(rect)
                    # Show inf count + finite stats if available
                    if not np.isnan(value) and std_values is not None:
                        text = f'∞ ({n_inf}/{n_total})\nμ={value:.2f}x ±{std_values[i, j]:.2f}'
                    else:
                        text = f'∞ ({n_inf}/{n_total})'
                    color = 'white'
                # Single-task mode with inf value
                elif np.isinf(value):
                    # Draw turquoise background for inf cells
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      facecolor='#00CED1', edgecolor='none')
                    ax.add_patch(rect)
                    text = '∞'
                    color = 'white'
                elif std_values is not None:
                    text = f'{value:.2f}x\n±{std_values[i, j]:.2f}'
                    color = 'black' if 0.8 < value < 2.0 else 'white'
                elif show_losses:
                    m_loss = maml_losses[i, j]
                    b_loss = baseline_losses[i, j]
                    text = f'{value:.2f}x\n({m_loss:.1e} / {b_loss:.1e})'
                    color = 'black' if 0.8 < value < 2.0 else 'white'
                else:
                    text = f'{value:.2f}x'
                    color = 'black' if 0.8 < value < 2.0 else 'white'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_loss_ratio_heatmap(
    ratios: np.ndarray,
    k_values: List[int],
    noise_levels: List[float],
    fixed_step: int,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    std_values: Optional[np.ndarray] = None,
    inf_counts: Optional[np.ndarray] = None,
    n_total: Optional[int] = None,
) -> plt.Figure:
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

    ratios = np.array(ratios, dtype=float)
    mask = np.isnan(ratios)

    # Diverging colormap centered at 1.0
    im = ax.imshow(ratios, cmap='RdYlGn_r', aspect='auto', vmin=0.2, vmax=1.8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'Loss Ratio @ step {fixed_step} (<1 = MAML better)')

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(noise_levels)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticklabels([f'{n:.0%}' for n in noise_levels])
    ax.set_xlabel('K (support set size)')
    ax.set_ylabel('Noise Level')

    has_inf_counts = inf_counts is not None

    for i in range(len(noise_levels)):
        for j in range(len(k_values)):
            if mask[i, j]:
                text = 'N/A'
                color = 'gray'
            else:
                value = ratios[i, j]
                n_inf = int(inf_counts[i, j]) if has_inf_counts else 0

                # Aggregated mode with some inf values
                if has_inf_counts and n_inf > 0:
                    # Draw turquoise background
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      facecolor='#00CED1', edgecolor='none')
                    ax.add_patch(rect)
                    # Show inf count + finite stats if available
                    if not np.isnan(value) and std_values is not None:
                        text = f'∞ ({n_inf}/{n_total})\nμ={value:.5f} ±{std_values[i, j]:.5f}'
                    else:
                        text = f'∞ ({n_inf}/{n_total})'
                    color = 'white'
                # Single-task mode with inf value
                elif np.isinf(value):
                    # Draw turquoise background for inf cells
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      facecolor='#00CED1', edgecolor='none')
                    ax.add_patch(rect)
                    text = '∞'
                    color = 'white'
                elif std_values is not None:
                    text = f'{value:.5f}\n±{std_values[i, j]:.5f}'
                    color = 'black' if 0.4 < value < 1.6 else 'white'
                else:
                    text = f'{value:.5f}'
                    color = 'black' if 0.4 < value < 1.6 else 'white'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_noise_robustness(
    noise_levels: List[float],
    maml_steps: List[float],
    baseline_steps: List[float],
    k_value: int,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[List[float]] = None,
    baseline_std: Optional[List[float]] = None,
) -> plt.Figure:
    """
    Graph 1: Noise robustness curves.

    Shows how noise affects time to reach plateau (steps_to_plateau).

    Args:
        noise_levels: X-axis values (noise percentages as decimals, e.g., 0.01 for 1%)
        maml_steps: Steps to plateau for MAML at each noise level
        baseline_steps: Steps to plateau for baseline at each noise level
        k_value: K value used (for title/labeling)
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
    maml_steps = np.array(maml_steps)
    baseline_steps = np.array(baseline_steps)

    ax.plot(noise_pct, maml_steps, 'b-o', label='MAML (θ*)', linewidth=2, markersize=8)
    ax.plot(noise_pct, baseline_steps, 'r-s', label='Baseline (θ₀)', linewidth=2, markersize=8)

    if maml_std is not None:
        maml_std = np.array(maml_std)
        ax.fill_between(noise_pct, maml_steps - maml_std, maml_steps + maml_std,
                        color='blue', alpha=0.2)
    if baseline_std is not None:
        baseline_std = np.array(baseline_std)
        ax.fill_between(noise_pct, baseline_steps - baseline_std, baseline_steps + baseline_std,
                        color='red', alpha=0.2)

    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Steps to Plateau')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_sample_efficiency(
    k_values: List[int],
    maml_losses: List[float],
    baseline_losses: List[float],
    noise_level: float,
    fixed_step: int,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
    maml_std: Optional[List[float]] = None,
    baseline_std: Optional[List[float]] = None,
) -> plt.Figure:
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

    maml_losses = np.array(maml_losses)
    baseline_losses = np.array(baseline_losses)

    ax.semilogy(k_values, maml_losses, 'b-o', label='MAML (θ*)', linewidth=2, markersize=8)
    ax.semilogy(k_values, baseline_losses, 'r-s', label='Baseline (θ₀)', linewidth=2, markersize=8)

    if maml_std is not None:
        maml_std = np.array(maml_std)
        ax.fill_between(k_values, maml_losses - maml_std, maml_losses + maml_std,
                        color='blue', alpha=0.2)
    if baseline_std is not None:
        baseline_std = np.array(baseline_std)
        ax.fill_between(k_values, baseline_losses - baseline_std, baseline_losses + baseline_std,
                        color='red', alpha=0.2)

    ax.set_xlabel('K (support set size)')
    ax.set_ylabel(f'Loss @ step {fixed_step}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale x-axis often useful for K values
    ax.set_xscale('log')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
