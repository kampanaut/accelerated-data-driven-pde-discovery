"""Intermediate types and helpers produced during evaluation.

These types live between the task-side `extract_coefficients` output and
the persisted schema in `src/evaluation/results.py`. `scripts/evaluate.py`
emits `MixerFineTuneOutput` per mixer from `fine_tune()`, then calls
`assemble_method_result()` to merge across mixers into a `MethodResult`
that gets serialized to `results.json` + `samples/*.npz`.

Keeping them in a separate file from `results.py` makes the persisted
schema the single source of truth for downstream consumers
(`visualize.py`, thesis plots), while the transient evaluation-loop
types + merge helper stay scoped to the evaluation pipeline.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from src.training.task_loader import CoefficientExtraction

if TYPE_CHECKING:
    from src.evaluation.results import MethodResult


@dataclass
class MixerFineTuneOutput:
    """Per-mixer output from one `fine_tune()` call.

    Produced for one (fast_model, mixer_idx, data) triple. `evaluate_task()`
    collects these (one per mixer) and merges the per-step extractions into
    the per-coefficient dict that `build_method_result()` consumes.

    Fields
    ------
    mixer_name : str
        The mixer's name as it appears in `task.mixer_names` — `"u"`, `"v"`,
        `"ω"`. Used by `evaluate_task` to prefix formula tags with the mixer
        when building merged path keys like `"u.jvp_lap_u"`.
    train_losses : np.ndarray
        Kendall-weighted total loss on the SUPPORT split at each inner step.
        Shape `(inner_steps + 1,)` — the extra entry at index 0 is the
        pre-adapt baseline snapshot before any optimizer step fires.
    holdout_losses : np.ndarray
        Same shape and semantics but evaluated on the HOLDOUT split — the
        generalization-facing snapshot at each inner step.
    per_step_extractions : Dict[int, Dict[str, Dict[str, CoefficientExtraction]]]
        Nested dict keyed first by the fixed_step index (e.g. 0, 16), then
        by coefficient name (e.g. "D_u", "k2"), then by formula tag (e.g.
        "jvp_lap_u", "neg1_minus_jvp_u"). Innermost values are
        `CoefficientExtraction` instances from the task's
        `extract_coefficients` call. For each fixed_step that's in the
        configured `fixed_steps` list, one snapshot entry exists here.
    pred_errors_per_step : Dict[int, np.ndarray]
        Per-point prediction residuals on the HOLDOUT split at each
        fixed_step: `pred - target[:, mixer_idx]`, shape `(holdout,)`.
        Used by the prediction-scatter visualization. One entry per
        fixed_step.
    """
    mixer_name: str
    train_losses: np.ndarray
    holdout_losses: np.ndarray
    per_step_extractions: Dict[int, Dict[str, Dict[str, CoefficientExtraction]]] = field(default_factory=dict)
    pred_errors_per_step: Dict[int, np.ndarray] = field(default_factory=dict)


def assemble_method_result(
    mixer_outputs: List[MixerFineTuneOutput],
    true_coefficients: Dict[str, float],
    fixed_steps: List[int],
) -> "MethodResult":
    """Merge per-mixer `MixerFineTuneOutput` list into a single `MethodResult`.

    Called by `evaluate_task()` once per combo per branch (maml/baseline),
    after collecting one `MixerFineTuneOutput` per mixer from `fine_tune()`.

    Four-step assembly:

    1. **FineTuneResult**: per-mixer loss trajectories keyed by mixer_name.

    2. **Cross-mixer merge** of per-step extractions. Each mixer's
       `per_step_extractions[step]` is `{coeff_name: {formula_tag: Extraction}}`.
       We prefix each `formula_tag` with the mixer's name to produce
       `path_key = f"{mixer_name}.{formula_tag}"`, and collect across
       mixers into one `{coeff_name: {path_key: PathCoefficientExtraction}}` dict per
       step. For coefficients that appear in multiple mixers (BR's k2,
       λ-ω's c), the merged dict has one entry per (mixer, formula)
       combination — downstream cross-path reconciliation in
       `build_method_result` computes `cross_path_mean` / `cross_path_std`
       over those entries.

    3. **Per-path raw values** for the NPZ side. For each path_key, stack
       the `CoefficientExtraction.values` tensors across fixed_steps into
       one `(n_fixed_steps, holdout)` array.

    4. **Prediction residuals** stacked into `(n_fixed_steps, holdout, n_outputs)`.
       Each mixer contributes one column in the final axis.

    Finally delegates to `build_method_result()` from `results.py`, which
    handles the within-step reconciliation (`cross_path_mean`,
    `cross_path_std`, `abs_error`, `pct_error`) and the per-step
    cross-coefficient `avg_error_pct_per_step` aggregate.

    Args:
        mixer_outputs: one `MixerFineTuneOutput` per mixer, in any order.
        true_coefficients: ground-truth values keyed by coefficient name,
            e.g. {"D_u": 2.58, "k1": 4.5, "k2": 11.63, "D_v": 13.22}. Names
            must match the outer keys produced by `task.extract_coefficients`.
        fixed_steps: the list of inner-loop step indices at which snapshots
            were taken. Sorted internally.

    Returns:
        A fully-populated `MethodResult` ready for insertion into
        `ComboResult.maml` or `ComboResult.baseline`.
    """
    # Local import to avoid a circular dependency: results.py doesn't
    # import eval_types, and eval_types imports build_method_result only
    # inside this function body at call time.
    from src.evaluation.results import (
        FineTuneResult,
        PathCoefficientExtraction,
        build_method_result,
    )

    sorted_fixed_steps = sorted(fixed_steps)

    # 1. Per-mixer loss trajectories → FineTuneResult
    fine_tune_result = FineTuneResult(
        per_mixer_train_losses={
            out.mixer_name: out.train_losses for out in mixer_outputs
        },
        per_mixer_holdout_losses={
            out.mixer_name: out.holdout_losses for out in mixer_outputs
        },
    )

    # 2. Cross-mixer merge of per-step extractions
    per_step_merged: List[Dict[str, Dict[str, PathCoefficientExtraction]]] = []
    for step in sorted_fixed_steps:
        merged_for_step: Dict[str, Dict[str, PathCoefficientExtraction]] = {}
        for mixer_output in mixer_outputs:
            assert step in mixer_output.per_step_extractions, (
                f"mixer '{mixer_output.mixer_name}' missing extraction snapshot "
                f"at fixed_step={step}. fine_tune() should have recorded it — "
                f"this indicates a programming error in the evaluation loop."
            )
            step_extractions = mixer_output.per_step_extractions[step]
            for coeff_name, formulas in step_extractions.items():
                if coeff_name not in merged_for_step:
                    merged_for_step[coeff_name] = {}
                for formula_tag, extraction in formulas.items():
                    path_key = f"{mixer_output.mixer_name}.{formula_tag}"
                    x = extraction.regressor
                    y = extraction.values * extraction.regressor
                    sl = float(
                        (x * y).sum()
                        / ((x * x).sum() + 1e-30)
                    )
                    ss_res = float(
                        ((y - sl * x) ** 2).sum()
                    )
                    ss_tot = float(
                        ((y - y.mean()) ** 2).sum()
                    )
                    r2 = (
                        1.0 - (ss_res / ss_tot)
                        if ss_tot > 0
                        else 0.0
                    )
                    merged_for_step[coeff_name][path_key] = PathCoefficientExtraction(
                        mean=float(extraction.mean.item()),
                        std=float(extraction.std.item()),
                        regressor_name=extraction.regressor_name,
                        r2=r2
                    )
        per_step_merged.append(merged_for_step)

    # 3. Per-path raw values + regressors: (n_fixed_steps, holdout) per path_key
    per_path_raw_values: Dict[str, np.ndarray] = {}
    per_path_regressor_values: Dict[str, np.ndarray] = {}
    for mixer_output in mixer_outputs:
        mixer_name = mixer_output.mixer_name
        first_step = sorted_fixed_steps[0]
        first_step_extractions = mixer_output.per_step_extractions[first_step]
        for coeff_name, formulas in first_step_extractions.items():
            for formula_tag in formulas.keys():
                path_key = f"{mixer_name}.{formula_tag}"
                val_arrs = []
                reg_arrs = []
                for step in sorted_fixed_steps:
                    ext = mixer_output.per_step_extractions[step][coeff_name][formula_tag]
                    val_arrs.append(ext.values.detach().cpu().numpy())
                    reg_arrs.append(ext.regressor.detach().cpu().numpy())
                per_path_raw_values[path_key] = np.stack(val_arrs, axis=0)
                per_path_regressor_values[path_key] = np.stack(reg_arrs, axis=0)

    # 4. Prediction residuals: (n_fixed_steps, holdout, n_outputs)
    pred_errors_stacked: Optional[np.ndarray] = None
    if all(out.pred_errors_per_step for out in mixer_outputs):
        per_step_stacked = []
        for step in sorted_fixed_steps:
            # Stack per-mixer (holdout,) arrays as columns → (holdout, n_outputs)
            cols = [out.pred_errors_per_step[step] for out in mixer_outputs]
            per_step_stacked.append(np.stack(cols, axis=-1))
        pred_errors_stacked = np.stack(per_step_stacked, axis=0)

    return build_method_result(
        fine_tune_result=fine_tune_result,
        true_coefficients=true_coefficients,
        per_step_extractions=per_step_merged,
        per_path_raw_values=per_path_raw_values,
        per_path_regressor_values=per_path_regressor_values,
        pred_snapshots=pred_errors_stacked,
    )
