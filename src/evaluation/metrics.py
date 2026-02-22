"""
Metric computation for MAML vs baseline comparison.

Primary metrics:
- Steps to plateau (primary metric for speedup) - detects longest flat region
- Speedup ratio (baseline_steps_to_plateau / maml_steps_to_plateau)
- Loss at plateau (quality of convergence)
- Loss at fixed step p
- Loss ratio at fixed step p

Legacy (deprecated):
- Steps to threshold (L* = 10^-6)
- Steps to lowest - replaced by plateau detection for monotonic curves
"""

from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def steps_to_threshold(
    losses: NDArray[np.floating[Any]], threshold: float = 1e-6
) -> Optional[int]:
    """
    Find first step where loss drops below threshold.

    Args:
        losses: Loss values at each gradient step [L_0, L_1, ..., L_n]
        threshold: Target loss value (L*)

    Returns:
        Step index where loss < threshold, or None if never reached
    """
    for step, loss in enumerate(losses):
        if loss < threshold:
            return step
    return None


def steps_to_lowest(losses: NDArray[np.floating[Any]]) -> int:
    """
    Find step with minimum loss value.

    Used as fallback metric when threshold is never reached.

    Args:
        losses: Loss values at each gradient step

    Returns:
        Step index with minimum loss
    """
    return int(min(range(len(losses)), key=lambda i: losses[i]))


def steps_to_plateau(
    losses: np.ndarray,
    deriv_threshold: float = 1e-3,
) -> int:
    """
    Find step where the longest plateau begins using relative derivative.

    Algorithm:
    1. Compute relative derivatives: (loss[i+1] - loss[i]) / loss[i]
    2. Find steps where |relative derivative| < threshold (flat regions)
    3. Group flat steps by adjacency into contiguous runs
    4. Select the longest run as the plateau
    5. Return the start of that plateau

    Args:
        losses: Loss values at each gradient step
        deriv_threshold: Maximum |relative derivative| to consider "flat"
                         (e.g., 1e-3 means <0.1% change per step)

    Returns:
        Step index where longest plateau begins, or steps_to_lowest if no plateau
    """
    if len(losses) < 2:
        return 0

    losses_arr = np.array(losses)

    # 1. Compute relative derivatives: (loss[i+1] - loss[i]) / loss[i]
    # Use loss[i] (current step) as denominator for normalization
    derivs = np.diff(losses_arr)
    # Avoid division by zero - use small epsilon for near-zero losses
    denominators = np.maximum(np.abs(losses_arr[:-1]), 1e-12)
    relative_derivs = derivs / denominators

    # 2. Find flat step indices where |relative derivative| < threshold
    flat_indices = np.where(np.abs(relative_derivs) < deriv_threshold)[0]

    if len(flat_indices) == 0:
        # No plateau detected - fall back to minimum
        return steps_to_lowest(losses)

    # 3. Group adjacent indices into contiguous runs
    # Find where gaps occur (diff > 1 means non-adjacent)
    gaps = np.diff(flat_indices) > 1
    run_starts = np.concatenate([[0], np.where(gaps)[0] + 1])
    run_ends = np.concatenate([np.where(gaps)[0] + 1, [len(flat_indices)]])
    run_lengths = run_ends - run_starts

    # 4. Find longest run, return its start
    longest_run_idx = np.argmax(run_lengths)
    return int(flat_indices[run_starts[longest_run_idx]])


def loss_at_step(losses: List[float], step: int) -> float:
    """
    Get loss value at a specific step.

    Args:
        losses: Loss values at each gradient step
        step: Step index to query

    Returns:
        Loss value at that step

    Raises:
        IndexError: If step >= len(losses)
    """
    return losses[step]


def speedup_ratio(
    maml_losses: NDArray[np.floating[Any]],
    baseline_losses: NDArray[np.floating[Any]],
    threshold: float = 1e-6,
) -> Tuple[Optional[float], str]:
    """
    Compute speedup ratio: baseline_steps / maml_steps.

    If neither method reaches threshold, uses steps-to-lowest as fallback.
    If only one reaches threshold, returns None (incomparable).

    Args:
        maml_losses: MAML fine-tuning loss curve
        baseline_losses: Baseline training loss curve
        threshold: L* threshold for convergence

    Returns:
        Tuple of (speedup_ratio, method_used)
        - speedup > 1 means MAML is faster
        - method_used is "threshold" or "lowest"
    """
    maml_steps = steps_to_threshold(maml_losses, threshold)
    baseline_steps = steps_to_threshold(baseline_losses, threshold)

    # Both reached threshold
    if maml_steps is not None and baseline_steps is not None:
        if maml_steps == 0:
            return float("inf"), "threshold"
        return baseline_steps / maml_steps, "threshold"

    # Neither reached threshold - use fallback
    if maml_steps is None and baseline_steps is None:
        maml_lowest = steps_to_lowest(maml_losses)
        baseline_lowest = steps_to_lowest(baseline_losses)
        if maml_lowest == 0:
            return float("inf"), "lowest"
        return baseline_lowest / maml_lowest, "lowest"

    # Only one reached - can't compare fairly
    return None, "incomparable"


@dataclass
class SpeedupResult:
    """Result of speedup computation using plateau detection."""

    ratio: float  # baseline_steps / maml_steps (>1 means MAML faster)
    maml_steps: int  # steps to reach MAML's plateau
    baseline_steps: int  # steps to reach baseline's plateau
    maml_loss: float  # MAML's loss at plateau start
    baseline_loss: float  # baseline's loss at plateau start


def speedup_ratio_dynamic(
    maml_losses: NDArray[np.floating[Any]],
    baseline_losses: NDArray[np.floating[Any]],
    deriv_threshold: float = 1e-3,
) -> SpeedupResult:
    """
    Compute speedup ratio using plateau detection as the primary metric.

    Uses steps_to_plateau to find where each curve stops improving meaningfully,
    which handles monotonically decreasing curves better than steps_to_lowest.

    Args:
        maml_losses: MAML fine-tuning loss curve
        baseline_losses: Baseline training loss curve
        deriv_threshold: Maximum |derivative| to consider "flat" for plateau detection

    Returns:
        SpeedupResult with ratio, steps, and loss values for both methods
    """
    maml_step = steps_to_plateau(maml_losses, deriv_threshold)
    baseline_step = steps_to_plateau(baseline_losses, deriv_threshold)

    maml_loss = maml_losses[maml_step]
    baseline_loss = baseline_losses[baseline_step]

    # Avoid division by zero
    if maml_step == 0:
        ratio = float("inf")
    else:
        ratio = baseline_step / maml_step

    return SpeedupResult(
        ratio=ratio,
        maml_steps=maml_step,
        baseline_steps=baseline_step,
        maml_loss=maml_loss,
        baseline_loss=baseline_loss,
    )


@dataclass
class ComparisonMetrics:
    """Metrics comparing MAML to baseline for a single (task, K, noise) run."""

    # Steps to convergence (train loss)
    maml_steps_to_plateau: int
    baseline_steps_to_plateau: int

    # Speedup (using plateau detection on train loss)
    speedup: float  # Always defined - uses steps_to_plateau

    # Loss values at plateau (train)
    maml_plateau_loss: float
    baseline_plateau_loss: float

    # Loss at fixed steps {step: loss}
    maml_loss_at_steps: Dict[int, float]
    baseline_loss_at_steps: Dict[int, float]

    # Loss ratios at fixed steps {step: maml/baseline}
    loss_ratios: Dict[int, float]

    # Holdout evaluation (generalization)
    maml_steps_to_plateau_holdout: Optional[int] = None
    baseline_steps_to_plateau_holdout: Optional[int] = None
    holdout_speedup: Optional[float] = None
    maml_plateau_loss_holdout: Optional[float] = None
    baseline_plateau_loss_holdout: Optional[float] = None


def compute_comparison_metrics(
    maml_losses: NDArray[np.floating[Any]],
    baseline_losses: NDArray[np.floating[Any]],
    fixed_steps: NDArray[np.integer[Any]],
    maml_holdout_losses: NDArray[np.floating[Any]],
    baseline_holdout_losses: NDArray[np.floating[Any]],
    deriv_threshold: float = 1e-3,
) -> ComparisonMetrics:
    """
    Compute all comparison metrics from loss curves.

    Args:
        maml_losses: MAML fine-tuning train loss curve
        baseline_losses: Baseline training loss curve
        threshold: L* threshold for convergence detection
        fixed_steps: Steps at which to record loss values (default: [50, 100, 200])
        deriv_threshold: Maximum |derivative| for plateau detection
        maml_holdout_losses: MAML holdout loss curve (optional)
        baseline_holdout_losses: Baseline holdout loss curve (optional)

    Returns:
        ComparisonMetrics dataclass with all computed metrics
    """

    # Plateau detection (primary metric) on train loss
    speedup_result = speedup_ratio_dynamic(
        maml_losses, baseline_losses, deriv_threshold
    )

    # Loss at fixed steps
    max_step = min(len(maml_losses), len(baseline_losses)) - 1
    maml_at_steps = {}
    baseline_at_steps = {}
    ratios = {}

    for step in fixed_steps:
        if step - 1 <= max_step:
            m_loss = maml_losses[step - 1]
            b_loss = baseline_losses[step - 1]
            maml_at_steps[step] = m_loss
            baseline_at_steps[step] = b_loss
            ratios[step] = m_loss / b_loss if b_loss > 0 else float("inf")

    # Holdout metrics (generalization)
    maml_plateau_holdout = None
    baseline_plateau_holdout = None
    holdout_speedup = None
    maml_loss_holdout = None
    baseline_loss_holdout = None

    if maml_holdout_losses is not None and baseline_holdout_losses is not None:
        holdout_result = speedup_ratio_dynamic(
            maml_holdout_losses, baseline_holdout_losses, deriv_threshold
        )
        maml_plateau_holdout = holdout_result.maml_steps
        baseline_plateau_holdout = holdout_result.baseline_steps
        holdout_speedup = holdout_result.ratio
        maml_loss_holdout = holdout_result.maml_loss
        baseline_loss_holdout = holdout_result.baseline_loss

    return ComparisonMetrics(
        maml_steps_to_plateau=speedup_result.maml_steps,
        baseline_steps_to_plateau=speedup_result.baseline_steps,
        speedup=speedup_result.ratio,
        maml_plateau_loss=speedup_result.maml_loss,
        baseline_plateau_loss=speedup_result.baseline_loss,
        maml_loss_at_steps=maml_at_steps,
        baseline_loss_at_steps=baseline_at_steps,
        loss_ratios=ratios,
        maml_steps_to_plateau_holdout=maml_plateau_holdout,
        baseline_steps_to_plateau_holdout=baseline_plateau_holdout,
        holdout_speedup=holdout_speedup,
        maml_plateau_loss_holdout=maml_loss_holdout,
        baseline_plateau_loss_holdout=baseline_loss_holdout,
    )
