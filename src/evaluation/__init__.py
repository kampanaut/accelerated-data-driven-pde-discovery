"""
Evaluation module for MAML experiment analysis.

Provides metric computation and visualization for comparing
MAML-initialized vs baseline PDE operator networks.
"""

from .metrics import (
    steps_to_threshold,
    steps_to_lowest,
    speedup_ratio,
    speedup_ratio_dynamic,
    SpeedupResult,
    loss_at_step,
    compute_comparison_metrics,
    ComparisonMetrics,
)
from .graphs import (
    plot_convergence,
    plot_speedup_heatmap,
    plot_loss_ratio_heatmap,
    plot_noise_robustness,
    plot_sample_efficiency,
)

__all__ = [
    # Metrics
    "steps_to_threshold",
    "steps_to_lowest",
    "speedup_ratio",
    "speedup_ratio_dynamic",
    "SpeedupResult",
    "loss_at_step",
    "compute_comparison_metrics",
    "ComparisonMetrics",
    # Graphs
    "plot_convergence",
    "plot_speedup_heatmap",
    "plot_loss_ratio_heatmap",
    "plot_noise_robustness",
    "plot_sample_efficiency",
]
