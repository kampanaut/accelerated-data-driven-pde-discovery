"""Typed result structures for mixer-method evaluation.

Dataclasses mirror the results.json + samples/*.npz output.
Produced by evaluate.py, consumed by visualize.py.

Serialization split:
- `to_json_dict()` → nested dict for results.json (scalars, loss trajectories,
  per-step recovery summaries with per-path means + cross-path reconciliation)
- `to_npz_dict()` → flat string-keyed dict for np.savez_compressed (per-mixer
  fine-tune arrays, per-path raw per-point values, prediction residuals)

results.json structure:
    EvaluationResults
    ├── experiment_name: str
    ├── timestamp: str
    ├── config: EvalConfig (fixed_steps, noise_levels, etc.)
    └── tasks: Dict[str, TaskResult]
        └── {task_name}: TaskResult
            ├── task_name: str
            ├── coefficients: Dict[str, float]       (ground-truth values)
            ├── ic_type: str
            ├── n_samples: int
            ├── worse: WorseFlags                     (union across combos)
            ├── combos: List[ComboResult]
            │   └── ComboResult
            │       ├── k: int
            │       ├── noise: float
            │       ├── maml: MethodResult            (θ* fine-tuned)
            │       │   ├── fine_tune: FineTuneResult
            │       │   │   ├── per_mixer_train_losses: Dict[mixer_name, ndarray]
            │       │   │   └── per_mixer_holdout_losses: Dict[mixer_name, ndarray]
            │       │   ├── coefficient_recovery: SnapshotSummary
            │       │   │   ├── coefficients: Dict[coeff_name, CoefficientSnapshot]
            │       │   │   │   └── CoefficientSnapshot
            │       │   │   │       ├── true_value: float
            │       │   │   │       └── per_step: List[PerStepRecovery]  (one per fixed_step)
            │       │   │   │           └── PerStepRecovery
            │       │   │   │               ├── recoveries: Dict[path_key, RecoveryPath]
            │       │   │   │               │   └── RecoveryPath(mean, std)
            │       │   │   │               ├── cross_path_mean: float
            │       │   │   │               ├── cross_path_std: float
            │       │   │   │               ├── abs_error: float
            │       │   │   │               └── pct_error: float
            │       │   │   └── avg_error_pct_per_step: List[float]  (per-step aggregate)
            │       │   ├── per_path_raw_values: Dict[path_key, ndarray]  (NPZ only)
            │       │   └── pred_errors: ndarray              (NPZ only)
            │       ├── baseline: MethodResult         (θ₀ fine-tuned, same shape)
            │       └── worse: WorseFlags
            └── best_combo: BestComboData

Path keys are formed as `{mixer_name}.{formula_tag}` — e.g. `"u.jvp_lap_u"`
for mixer_u's D_u extraction via the laplacian JVP formula, or
`"v.neg1_minus_jvp_u"` for mixer_v's k2 extraction. This encoding lets
the visualizer parse each key to color by mixer or marker-by-formula.

For single-path coefficients (BR's D_u, lives only in mixer_u) the
`recoveries` dict has one entry and `cross_path_std = 0.0`. For multi-path
coefficients (BR's k2, λ-ω's c) multiple entries sit side-by-side and
`cross_path_std` measures inter-mixer agreement.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── Leaf structures ──────────────────────────────────────────────────────


@dataclass
class FineTuneResult:
    """Per-mixer train and holdout loss trajectories from inner solving.

    Each mixer has its own L-BFGS inner solve with its own trajectory, so the
    loss curves are stored per-mixer (keyed by mixer name: "u", "v", "ω").
    For `n_outputs=1` PDEs (NLHeat, Heat, NS-vorticity) the dicts have a single
    entry. For 2-output PDEs (BR, FHN, λ-ω) they have two entries.
    """
    per_mixer_train_losses: Dict[str, np.ndarray] = field(default_factory=dict)
    per_mixer_holdout_losses: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RecoveryPath:
    """One recovery path's estimate from the holdout batch.

    `mean` is the mean across collocation points of the per-point extracted
    value (e.g. mean of `jvp_wrt_uxxuyy` across the holdout for BR's D_u).
    `std` is the dispersion across those same points — the "within-path"
    consistency. A small std means the mixer is near-linear on that feature
    and the extraction is clean; a large std means the per-point estimates
    vary a lot across the holdout.
    """
    mean: float = 0.0
    std: float = 0.0
    regressor_name: str = ""


@dataclass
class PerStepRecovery:
    """All recovery paths for one coefficient at one fixed_step, with reconciliation.

    `recoveries` is keyed by path name like "u.jvp_lap_u" or "v.neg1_minus_jvp_u".
    Keys encode the mixer name + the extraction formula tag so the visualizer
    can parse them to color by mixer or marker-by-formula.

    Cross-path statistics are unweighted over `recoveries[*].mean`:
    - `cross_path_mean`: unweighted mean of per-path means. For a single-path
       coefficient, this is just that one path's mean.
    - `cross_path_std`: std of the per-path means. Zero when there's one path;
       for multi-path coefficients (BR's k2, λ-ω's c) it measures inter-mixer
       agreement — small std means the mixers reconcile cleanly, large std
       means they've absorbed the coefficient into different structures.

    `abs_error` is `|cross_path_mean - CoefficientSnapshot.true_value|` —
    the final reconciled recovery error for this coefficient at this step.

    `pct_error` is the same quantity normalized by |true_value|: `100 * abs_error
    / |true_value|`. Makes errors comparable across coefficients of different
    magnitudes (a 0.18 abs error on D_u=2.58 is ~7%, same abs error on k1=4.5
    is ~4%). Set to 0.0 when `true_value == 0` to avoid division by zero.
    """
    recoveries: Dict[str, RecoveryPath] = field(default_factory=dict)
    cross_path_mean: float = 0.0
    cross_path_std: float = 0.0
    abs_error: float = 0.0
    pct_error: float = 0.0


@dataclass
class CoefficientSnapshot:
    """Per-coefficient recovery summary across fixed_steps.

    `per_step[i]` corresponds to `EvalConfig.fixed_steps[i]`. For the typical
    `fixed_steps: [0, 16]`, index 0 is the pre-adapt snapshot, index 1 is
    the post-adapt snapshot. Adaptation lift for this coefficient is
    `per_step[0].abs_error - per_step[-1].abs_error`.
    """
    true_value: float = 0.0
    per_step: List[PerStepRecovery] = field(default_factory=list)


@dataclass
class SnapshotSummary:
    """Coefficient recovery for one method (MAML or baseline) at one combo."""
    coefficients: Dict[str, CoefficientSnapshot] = field(default_factory=dict)
    # Per-step aggregate: at each fixed_step, the mean of pct_error across all
    # coefficients in `coefficients`. One "headline number per step" for this
    # method at this combo, used for top-level convergence curves that don't
    # want to pick an individual coefficient.
    avg_error_pct_per_step: List[float] = field(default_factory=list)


@dataclass
class MethodResult:
    """Fine-tuning + coefficient recovery for one adapted model.

    One instance per (ComboResult.maml, ComboResult.baseline) — the two sit
    side-by-side in ComboResult carrying the θ* and θ₀ branches of the
    comparison.

    JSON-serialized: `fine_tune` loss trajectories, `coefficient_recovery`
    with per-step per-path reconciliation.

    NPZ-only: `per_path_raw_values` holds the raw per-point extracted values
    for each recovery path at each fixed_step — too big for JSON, useful for
    diagnostic scatter plots. `pred_errors` holds the prediction residuals
    for the prediction-scatter visualization.
    """
    fine_tune: FineTuneResult = field(default_factory=FineTuneResult)
    coefficient_recovery: SnapshotSummary = field(default_factory=SnapshotSummary)
    # NPZ-only fields — not in JSON
    per_path_raw_values: Dict[str, np.ndarray] = field(default_factory=dict)
    per_path_regressor_values: Dict[str, np.ndarray] = field(default_factory=dict)
    # both keyed "{mixer_name}.{formula_tag}" → shape (n_fixed_steps, holdout)
    pred_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    # shape (n_fixed_steps, holdout, n_outputs)


@dataclass
class WorseFlags:
    """MAML-worse-than-baseline flags at specific steps."""
    loss_steps: List[int] = field(default_factory=list)
    coeff_steps: Dict[str, List[int]] = field(default_factory=dict)


# ── Combo level ──────────────────────────────────────────────────────────


@dataclass
class ComboResult:
    """Result for one (k, noise) combination. Always complete — errors exit early."""
    k: int = 0
    noise: float = 0.0
    maml: MethodResult = field(default_factory=MethodResult)
    baseline: MethodResult = field(default_factory=MethodResult)
    worse: WorseFlags = field(default_factory=WorseFlags)

    @property
    def combo_key(self) -> str:
        return f"k_{self.k}_noise_{self.noise:.2f}"


# ── Best combo prediction ────────────────────────────────────────────────


@dataclass
class BestComboData:
    """Prediction snapshots from re-fine-tuning the best combo (for spatial viz)."""
    combo_key: str = ""
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))   # (n_steps, holdout, n_outputs)
    true_targets: np.ndarray = field(default_factory=lambda: np.array([]))  # (holdout, n_outputs)
    x_pts: np.ndarray = field(default_factory=lambda: np.array([]))         # (holdout,)
    y_pts: np.ndarray = field(default_factory=lambda: np.array([]))         # (holdout,)
    steps: np.ndarray = field(default_factory=lambda: np.array([]))         # (n_steps,)
    coeff_error: np.ndarray = field(default_factory=lambda: np.array([]))   # (n_steps,)


# ── Task level ───────────────────────────────────────────────────────────


@dataclass
class TaskResult:
    """Full evaluation result for one test task."""
    task_name: str = ""
    coefficients: Dict[str, float] = field(default_factory=dict)
    ic_type: str = ""
    n_samples: int = 0
    worse: WorseFlags = field(default_factory=WorseFlags)  # union across all combos
    combos: List[ComboResult] = field(default_factory=list)
    best_combo: BestComboData = field(default_factory=BestComboData)

    def combo_by_key(self, combo_key: str) -> ComboResult:
        """Look up a combo by its string key (e.g. 'k_800_noise_0.00')."""
        for c in self.combos:
            if c.combo_key == combo_key:
                return c
        raise KeyError(f"No combo with key {combo_key}")

    @classmethod
    def from_json_and_npz(
        cls, task_dict: Dict[str, Any], raw_npz: Dict[str, Any]
    ) -> "TaskResult":
        """Reconstruct TaskResult from results.json task entry + NPZ arrays.

        Inverse of `to_json_dict` + `to_npz_dict`. Used by visualize.py to
        load evaluation results into typed dataclass instances. Scalars come
        from the JSON task_dict; per-mixer loss arrays and per-path raw
        values come from raw_npz.
        """
        combos: List[ComboResult] = []
        for combo_dict in task_dict.get("combos", []):
            k = int(combo_dict.get("k", 0))
            noise = float(combo_dict.get("noise", 0.0))
            ck = f"k_{k}_noise_{noise:.2f}"

            methods: Dict[str, MethodResult] = {}
            for label in ("maml", "baseline"):
                method_dict = combo_dict.get(label, {})

                # Per-mixer fine-tune losses — prefer NPZ, fall back to JSON list
                per_mixer_train: Dict[str, np.ndarray] = {}
                per_mixer_holdout: Dict[str, np.ndarray] = {}
                ft_prefix = f"{ck}/{label}/fine_tune/"
                for key, arr in raw_npz.items():
                    if not key.startswith(ft_prefix):
                        continue
                    rest = key[len(ft_prefix):].split("/")
                    if len(rest) != 2:
                        continue
                    mname, subfield = rest
                    if subfield == "train_losses":
                        per_mixer_train[mname] = arr
                    elif subfield == "holdout_losses":
                        per_mixer_holdout[mname] = arr

                if not per_mixer_train or not per_mixer_holdout:
                    ft_json = method_dict.get("fine_tune", {})
                    for mname, lst in ft_json.get("per_mixer_train_losses", {}).items():
                        per_mixer_train.setdefault(mname, np.asarray(lst))
                    for mname, lst in ft_json.get("per_mixer_holdout_losses", {}).items():
                        per_mixer_holdout.setdefault(mname, np.asarray(lst))

                fine_tune = FineTuneResult(
                    per_mixer_train_losses=per_mixer_train,
                    per_mixer_holdout_losses=per_mixer_holdout,
                )

                # Coefficient recovery from JSON
                summary = _summary_from_json(method_dict.get("coefficient_recovery", {}))

                # Per-path raw values + regressors from NPZ
                per_path_raw_values: Dict[str, np.ndarray] = {}
                per_path_regressor_values: Dict[str, np.ndarray] = {}
                raw_prefix = f"{ck}/{label}/raw_values/"
                reg_prefix = f"{ck}/{label}/regressors/"
                for key, arr in raw_npz.items():
                    if key.startswith(raw_prefix):
                        per_path_raw_values[key[len(raw_prefix):]] = arr
                    elif key.startswith(reg_prefix):
                        per_path_regressor_values[key[len(reg_prefix):]] = arr

                pred_errors = raw_npz.get(
                    f"{ck}/{label}/pred_errors", np.array([])
                )

                methods[label] = MethodResult(
                    fine_tune=fine_tune,
                    coefficient_recovery=summary,
                    per_path_raw_values=per_path_raw_values,
                    per_path_regressor_values=per_path_regressor_values,
                    pred_errors=pred_errors,
                )

            worse_dict = combo_dict.get("worse", {})
            worse = WorseFlags(
                loss_steps=worse_dict.get("loss_steps", []),
                coeff_steps=worse_dict.get("coeff_steps", {}),
            )

            combos.append(ComboResult(
                k=k, noise=noise,
                maml=methods["maml"],
                baseline=methods["baseline"],
                worse=worse,
            ))

        task_worse_dict = task_dict.get("worse", {})
        task_worse = WorseFlags(
            loss_steps=task_worse_dict.get("loss_steps", []),
            coeff_steps=task_worse_dict.get("coeff_steps", {}),
        )

        bc = BestComboData()
        bc_key = raw_npz.get("best_combo/key")
        if bc_key is not None:
            bc = BestComboData(
                combo_key=str(bc_key),
                predictions=raw_npz.get("best_combo/predictions", np.array([])),
                true_targets=raw_npz.get("best_combo/true_targets", np.array([])),
                x_pts=raw_npz.get("best_combo/x_pts", np.array([])),
                y_pts=raw_npz.get("best_combo/y_pts", np.array([])),
                steps=raw_npz.get("best_combo/steps", np.array([])),
                coeff_error=raw_npz.get("best_combo/coeff_error", np.array([])),
            )

        return cls(
            task_name=task_dict.get("task_name", ""),
            coefficients=task_dict.get("coefficients", {}),
            ic_type=task_dict.get("ic_type", ""),
            n_samples=task_dict.get("n_samples", 0),
            worse=task_worse,
            combos=combos,
            best_combo=bc,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to results.json format (new nested schema)."""
        return {
            "task_name": self.task_name,
            "coefficients": self.coefficients,
            "ic_type": self.ic_type,
            "n_samples": self.n_samples,
            "worse": {
                "loss_steps": self.worse.loss_steps,
                "coeff_steps": self.worse.coeff_steps,
            },
            "combos": [_combo_to_json(c) for c in self.combos],
        }

    def to_npz_dict(self) -> Dict[str, Any]:
        """Serialize to flat NPZ key→array dict.

        Key layout:
            {combo_key}/{label}/fine_tune/{mixer}/train_losses
            {combo_key}/{label}/fine_tune/{mixer}/holdout_losses
            {combo_key}/{label}/raw_values/{path_key}      # (n_fixed_steps, holdout)
            {combo_key}/{label}/pred_errors                 # (n_fixed_steps, holdout, n_outputs)
            best_combo/*
        """
        npz: Dict[str, Any] = {}

        for combo in self.combos:
            ck = combo.combo_key
            for label, method in [("maml", combo.maml), ("baseline", combo.baseline)]:
                # Per-mixer fine-tune trajectories
                for mname, arr in method.fine_tune.per_mixer_train_losses.items():
                    npz[f"{ck}/{label}/fine_tune/{mname}/train_losses"] = arr
                for mname, arr in method.fine_tune.per_mixer_holdout_losses.items():
                    npz[f"{ck}/{label}/fine_tune/{mname}/holdout_losses"] = arr

                # Per-path raw values + regressors
                for path_key, arr in method.per_path_raw_values.items():
                    npz[f"{ck}/{label}/raw_values/{path_key}"] = arr
                for path_key, arr in method.per_path_regressor_values.items():
                    npz[f"{ck}/{label}/regressors/{path_key}"] = arr

                # Prediction residuals
                if method.pred_errors.size > 0:
                    npz[f"{ck}/{label}/pred_errors"] = method.pred_errors

        # Best combo prediction data (unchanged from old schema)
        if self.best_combo.combo_key:
            bc = self.best_combo
            npz["best_combo/key"] = np.array(bc.combo_key)
            if bc.predictions.size > 0:
                npz["best_combo/predictions"] = bc.predictions
            if bc.true_targets.size > 0:
                npz["best_combo/true_targets"] = bc.true_targets
            if bc.x_pts.size > 0:
                npz["best_combo/x_pts"] = bc.x_pts
            if bc.y_pts.size > 0:
                npz["best_combo/y_pts"] = bc.y_pts
            if bc.steps.size > 0:
                npz["best_combo/steps"] = bc.steps
            if bc.coeff_error.size > 0:
                npz["best_combo/coeff_error"] = bc.coeff_error

        return npz


# ── Top level ────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """Snapshot of evaluation parameters."""
    k_values: List[int] = field(default_factory=list)
    noise_levels: List[float] = field(default_factory=list)
    fine_tune_lr: float = 0.01
    max_steps: int = 50
    threshold: float = 0.0005
    fixed_steps: List[int] = field(default_factory=list)
    holdout_size: int = 5000
    pde_type: str = ""


@dataclass
class EvaluationResults:
    """Top-level evaluation output — serializes to results.json."""
    experiment_name: str = ""
    timestamp: str = ""
    config: EvalConfig = field(default_factory=EvalConfig)
    tasks: Dict[str, TaskResult] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, eval_dir: Path) -> "EvaluationResults":
        """Load evaluation results from an evaluation/ directory.

        Reads eval_dir/results.json for metadata + coefficient recovery.
        Reads eval_dir/samples/{task}.npz for raw arrays (if present).
        """
        import json

        results_path = eval_dir / "results.json"
        with open(results_path) as f:
            raw = json.load(f)

        config_dict = raw.get("config", {})
        config = EvalConfig(
            k_values=config_dict.get("k_values", []),
            noise_levels=config_dict.get("noise_levels", []),
            fine_tune_lr=config_dict.get("fine_tune_lr", 0.01),
            max_steps=config_dict.get("max_steps", 50),
            threshold=config_dict.get("threshold", 0.0005),
            fixed_steps=config_dict.get("fixed_steps", []),
            holdout_size=config_dict.get("holdout_size", 5000),
            pde_type=config_dict.get("pde_type", ""),
        )

        samples_dir = eval_dir / "samples"
        tasks: Dict[str, TaskResult] = {}
        for task_name, task_dict in raw.get("tasks", {}).items():
            npz_path = samples_dir / f"{task_name}.npz"
            raw_npz = dict(np.load(npz_path)) if npz_path.exists() else {}
            tasks[task_name] = TaskResult.from_json_and_npz(task_dict, raw_npz)

        return cls(
            experiment_name=raw.get("experiment_name", ""),
            timestamp=raw.get("timestamp", ""),
            config=config,
            tasks=tasks,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to results.json format.

        `fixed_steps` lives at the top level under `config`, not duplicated
        into each combo — the per-combo `per_step` lists in each
        CoefficientSnapshot are ordered by `config.fixed_steps` so readers
        can line them up positionally.
        """
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "tasks": {
                name: tr.to_json_dict() for name, tr in self.tasks.items()
            },
        }


# ── Helpers ──────────────────────────────────────────────────────────────


def build_method_result(
    fine_tune_result: FineTuneResult,
    true_coefficients: Dict[str, float],
    per_step_extractions: List[Dict[str, Dict[str, RecoveryPath]]],
    per_path_raw_values: Dict[str, np.ndarray],
    per_path_regressor_values: Optional[Dict[str, np.ndarray]] = None,
    pred_snapshots: Optional[np.ndarray] = None,
) -> MethodResult:
    """Build a MethodResult from per-mixer fine-tune output + per-step extractions.

    Args:
        fine_tune_result: per-mixer train/holdout loss trajectories
        true_coefficients: ground-truth values from the task, e.g.
            {"D_u": 2.58, "k1": 4.5, "k2": 11.63, "D_v": 13.22}
        per_step_extractions: list of length len(fixed_steps). Entry `i` is a
            dict {coeff_name: {path_key: RecoveryPath}} — the extracted
            recoveries from all mixers for the i-th fixed_step. The caller
            builds this by merging task.extract_coefficients(...) outputs
            across mixers, keying each path by "{mixer_name}.{formula_tag}".
        per_path_raw_values: NPZ-only. {path_key: (n_fixed_steps, holdout)}.
        pred_snapshots: NPZ-only. (n_fixed_steps, holdout, n_outputs).

    Cross-path reconciliation (cross_path_mean, cross_path_std, abs_error,
    pct_error) is computed here from the per-path means in each
    PerStepRecovery, then SnapshotSummary.avg_error_pct_per_step is
    aggregated as the mean of per-coefficient pct_error across all
    coefficients at each step.
    """
    # Collect all coefficient names that appeared at any step
    all_coeff_names: set[str] = set()
    for step_data in per_step_extractions:
        all_coeff_names.update(step_data.keys())

    coefficients: Dict[str, CoefficientSnapshot] = {}
    for name in sorted(all_coeff_names):
        true_val = float(true_coefficients.get(name, 0.0))
        per_step_list: List[PerStepRecovery] = []
        for step_data in per_step_extractions:
            paths = step_data.get(name, {})
            path_means = [p.mean for p in paths.values()]
            if path_means:
                cross_mean = float(np.mean(path_means))
                cross_std = float(np.std(path_means)) if len(path_means) > 1 else 0.0
                abs_err = abs(cross_mean - true_val)
                pct_err = (100.0 * abs_err / abs(true_val)) if true_val != 0.0 else 0.0
            else:
                cross_mean = 0.0
                cross_std = 0.0
                abs_err = 0.0
                pct_err = 0.0
            per_step_list.append(PerStepRecovery(
                recoveries=dict(paths),
                cross_path_mean=cross_mean,
                cross_path_std=cross_std,
                abs_error=abs_err,
                pct_error=pct_err,
            ))
        coefficients[name] = CoefficientSnapshot(
            true_value=true_val,
            per_step=per_step_list,
        )

    # Aggregate pct_error per step: mean across all coefficients at that step
    n_steps = len(per_step_extractions)
    avg_error_pct: List[float] = []
    for i in range(n_steps):
        errs = [
            snap.per_step[i].pct_error
            for snap in coefficients.values()
            if i < len(snap.per_step)
        ]
        avg_error_pct.append(float(np.mean(errs)) if errs else 0.0)

    summary = SnapshotSummary(
        coefficients=coefficients,
        avg_error_pct_per_step=avg_error_pct,
    )

    return MethodResult(
        fine_tune=fine_tune_result,
        coefficient_recovery=summary,
        per_path_raw_values=per_path_raw_values,
        per_path_regressor_values=per_path_regressor_values or {},
        pred_errors=pred_snapshots if pred_snapshots is not None else np.array([]),
    )


def _combo_to_json(combo: ComboResult) -> Dict[str, Any]:
    """Serialize one ComboResult to a JSON-friendly nested dict.

    Output shape: `{k, noise, maml: MethodResult_json, baseline: MethodResult_json, worse}`
    where each `MethodResult_json` carries its `fine_tune` loss lists and
    its nested `coefficient_recovery` (see `_summary_to_json`).
    """
    return {
        "k": combo.k,
        "noise": combo.noise,
        "maml": _method_to_json(combo.maml),
        "baseline": _method_to_json(combo.baseline),
        "worse": {
            "loss_steps": combo.worse.loss_steps,
            "coeff_steps": combo.worse.coeff_steps,
        },
    }


def _method_to_json(method: MethodResult) -> Dict[str, Any]:
    """Serialize one MethodResult (JSON-only fields — NPZ fields are excluded).

    `per_path_raw_values` and `pred_errors` are NPZ-only and NOT emitted here.
    """
    return {
        "fine_tune": {
            "per_mixer_train_losses": {
                mname: arr.tolist() if hasattr(arr, "tolist") else list(arr)
                for mname, arr in method.fine_tune.per_mixer_train_losses.items()
            },
            "per_mixer_holdout_losses": {
                mname: arr.tolist() if hasattr(arr, "tolist") else list(arr)
                for mname, arr in method.fine_tune.per_mixer_holdout_losses.items()
            },
        },
        "coefficient_recovery": _summary_to_json(method.coefficient_recovery),
    }


def _summary_to_json(summary: SnapshotSummary) -> Dict[str, Any]:
    """Serialize a SnapshotSummary to a JSON-friendly nested dict."""
    coeffs: Dict[str, Any] = {}
    for name, snap in summary.coefficients.items():
        coeffs[name] = {
            "true_value": snap.true_value,
            "per_step": [
                {
                    "recoveries": {
                        path: {"mean": rp.mean, "std": rp.std, "regressor_name": rp.regressor_name}
                        for path, rp in ps.recoveries.items()
                    },
                    "cross_path_mean": ps.cross_path_mean,
                    "cross_path_std": ps.cross_path_std,
                    "abs_error": ps.abs_error,
                    "pct_error": ps.pct_error,
                }
                for ps in snap.per_step
            ],
        }
    return {
        "coefficients": coeffs,
        "avg_error_pct_per_step": summary.avg_error_pct_per_step,
    }


def _summary_from_json(d: Dict[str, Any]) -> SnapshotSummary:
    """Inverse of `_summary_to_json`."""
    coefficients: Dict[str, CoefficientSnapshot] = {}
    for name, snap_dict in d.get("coefficients", {}).items():
        per_step: List[PerStepRecovery] = []
        for ps_dict in snap_dict.get("per_step", []):
            recoveries = {
                path: RecoveryPath(mean=rp["mean"], std=rp["std"], regressor_name=rp.get("regressor_name", ""))
                for path, rp in ps_dict.get("recoveries", {}).items()
            }
            per_step.append(PerStepRecovery(
                recoveries=recoveries,
                cross_path_mean=ps_dict.get("cross_path_mean", 0.0),
                cross_path_std=ps_dict.get("cross_path_std", 0.0),
                abs_error=ps_dict.get("abs_error", 0.0),
                pct_error=ps_dict.get("pct_error", 0.0),
            ))
        coefficients[name] = CoefficientSnapshot(
            true_value=snap_dict.get("true_value", 0.0),
            per_step=per_step,
        )
    return SnapshotSummary(
        coefficients=coefficients,
        avg_error_pct_per_step=d.get("avg_error_pct_per_step", []),
    )
