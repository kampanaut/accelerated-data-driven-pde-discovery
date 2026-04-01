"""Typed result structures for MAML evaluation.

Dataclasses mirror the results.json + samples/*.npz output.
Produced by evaluate.py, consumed by visualize.py.

Serialization:
- to_json_dict() → nested dict for results.json
- to_npz_dict() → flat string-keyed dict for np.savez_compressed

results.json structure:
    EvaluationResults
    ├── experiment_name: str
    ├── timestamp: str
    ├── config: EvalConfig
    │   ├── k_values: List[int]
    │   ├── noise_levels: List[float]
    │   ├── fine_tune_lr: float
    │   ├── max_steps: int
    │   ├── threshold: float
    │   ├── fixed_steps: List[int]
    │   ├── holdout_size: int
    │   └── pde_type: str
    │
    └── tasks: Dict[str, TaskResult]
        └── {task_name}: TaskResult
            ├── task_name: str
            ├── coefficients: Dict[str, float]
            ├── coefficient_specs: List[CoefficientSpec]
            ├── ic_type: str
            ├── n_samples: int
            ├── loss_worse_steps: List[int]
            ├── coeff_worse_steps: Dict[str, List[int]]
            │
            └── combos: List[ComboResult]
                └── ComboResult
                    ├── k: int
                    ├── noise: float
                    ├── error: Optional[str]
                    ├── maml: MethodResult
                    │   ├── fine_tune: FineTuneResult
                    │   │   ├── train_losses: List[float]
                    │   │   └── holdout_losses: List[float]
                    │   └── coefficient_recovery: SnapshotSummary
                    │       ├── error_pct: List[float]
                    │       └── coefficients: Dict[str, CoefficientSnapshot]
                    │           └── {name}: CoefficientSnapshot
                    │               ├── true_value: float
                    │               ├── recovered: List[float]
                    │               ├── error_pct: List[float]
                    │               ├── mean: List[float]
                    │               └── std: List[float]
                    ├── baseline: MethodResult (same shape)
                    └── worse: WorseFlags
                        ├── loss_steps: List[int]
                        └── coeff_steps: Dict[str, List[int]]
            │
            └── best_combo: BestComboData
                ├── combo_key: str
                ├── predictions: ndarray (n_steps, holdout, n_outputs)
                ├── true_targets: ndarray (holdout, n_outputs)
                ├── x_pts: ndarray (holdout,)
                ├── y_pts: ndarray (holdout,)
                ├── steps: ndarray (n_steps,)
                └── coeff_error: ndarray (n_steps,)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# ── Leaf structures ──────────────────────────────────────────────────────


@dataclass
class FineTuneResult:
    """Per-step train and holdout losses from fine-tuning."""
    train_losses: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    holdout_losses: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))


@dataclass
class CoefficientSnapshot:
    """Per-coefficient recovery summary across fixed steps."""
    true_value: float = 0.0
    recovered: List[float] = field(default_factory=list)
    error_pct: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)


@dataclass
class SnapshotSummary:
    """Coefficient recovery for one method (MAML or baseline) at one combo."""
    coefficients: Dict[str, CoefficientSnapshot] = field(default_factory=dict)
    error_pct: List[float] = field(default_factory=list)  # mean across all coeffs per step


@dataclass
class MethodResult:
    """Fine-tuning + coefficient recovery for one method."""
    fine_tune: FineTuneResult = field(default_factory=FineTuneResult)
    coefficient_recovery: SnapshotSummary = field(default_factory=SnapshotSummary)
    # Raw arrays for NPZ (not serialized to JSON)
    jacobian_estimates: Dict[str, np.ndarray] = field(default_factory=dict)  # coeff_name → (n_steps, holdout)
    jacobian_true: Dict[str, np.ndarray] = field(default_factory=dict)      # coeff_name → (1,)
    pred_errors: np.ndarray = field(default_factory=lambda: np.array([]))    # (n_steps, holdout, n_outputs)
    weights: np.ndarray = field(default_factory=lambda: np.array([]))        # (n_steps, n_params) if log_weights


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
    coefficient_specs: list = field(default_factory=list)
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

        Inverse of to_json_dict() + to_npz_dict(). Used by visualize.py
        to load evaluation results into typed structures.
        """
        coeff_names = [s["name"] for s in task_dict.get("coefficient_specs", [])]

        # Discover combo keys from NPZ prefixes AND JSON keys
        combo_keys: set[str] = set()
        for key in raw_npz:
            prefix = key.rsplit("/")[0]
            if prefix != "best_combo":
                combo_keys.add(prefix)
        # Also from JSON coefficient_recovery_ keys (for when NPZ is absent)
        for key in task_dict:
            if key.startswith("coefficient_recovery_"):
                combo_keys.add(key[len("coefficient_recovery_"):])

        combos: List[ComboResult] = []
        for ck in sorted(combo_keys):
            # Parse k and noise from combo_key: "k_800_noise_0.00"
            parts = ck.split("_")
            k = int(parts[1])
            noise = float(parts[3])

            methods: Dict[str, MethodResult] = {}
            for label in ("maml", "baseline"):
                train_arr = raw_npz.get(f"{ck}/{label}_train_losses")
                holdout_arr = raw_npz.get(f"{ck}/{label}_holdout_losses")

                ft = FineTuneResult(
                    train_losses=train_arr if train_arr is not None else np.array([]),
                    holdout_losses=holdout_arr if holdout_arr is not None else np.array([]),
                )

                # Jacobian arrays
                jac_estimates: Dict[str, np.ndarray] = {}
                jac_true: Dict[str, np.ndarray] = {}
                for name in coeff_names:
                    est = raw_npz.get(f"{ck}/{label}/{name}")
                    if est is not None:
                        jac_estimates[name] = est
                    true_val = raw_npz.get(f"{ck}/{label}/{name}_true")
                    if true_val is not None:
                        jac_true[name] = true_val

                pred_err = raw_npz.get(f"{ck}/{label}/pred_errors")
                weights = raw_npz.get(f"{ck}/{label}/weights")

                # Coefficient recovery from JSON
                recovery_dict = task_dict.get(f"coefficient_recovery_{ck}", {})
                label_recovery = recovery_dict.get(label, {})
                summary = _flat_to_summary(label_recovery, coeff_names)

                methods[label] = MethodResult(
                    fine_tune=ft,
                    coefficient_recovery=summary,
                    jacobian_estimates=jac_estimates,
                    jacobian_true=jac_true,
                    pred_errors=pred_err if pred_err is not None else np.array([]),
                    weights=weights if weights is not None else np.array([]),
                )

            worse_dict = task_dict.get(f"worse_{ck}", {})
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

        task_worse = WorseFlags(
            loss_steps=task_dict.get("loss_worse_steps", []),
            coeff_steps=task_dict.get("coeff_worse_steps", {}),
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
            coefficient_specs=task_dict.get("coefficient_specs", []),
            ic_type=task_dict.get("ic_type", ""),
            n_samples=task_dict.get("n_samples", 0),
            worse=task_worse,
            combos=combos,
            best_combo=bc,
        )

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize to results.json format."""
        d: Dict[str, Any] = {
            "task_name": self.task_name,
            "coefficients": self.coefficients,
            "coefficient_specs": self.coefficient_specs,
            "ic_type": self.ic_type,
            "n_samples": self.n_samples,
            "loss_worse_steps": self.worse.loss_steps,
            "coeff_worse_steps": self.worse.coeff_steps,
        }

        for combo in self.combos:
            ck = combo.combo_key

            d[f"coefficient_recovery_{ck}"] = {
                "fixed_steps": None,  # filled by caller from EvalConfig
                "maml": _summary_to_flat(combo.maml.coefficient_recovery),
                "baseline": _summary_to_flat(combo.baseline.coefficient_recovery),
            }

            d[f"worse_{ck}"] = {
                "loss_steps": combo.worse.loss_steps,
                "coeff_steps": combo.worse.coeff_steps,
            }

        return d

    def to_npz_dict(self) -> Dict[str, Any]:
        """Serialize to flat NPZ key→array dict."""
        npz: Dict[str, Any] = {}

        for combo in self.combos:
            ck = combo.combo_key

            for label, method in [("maml", combo.maml), ("baseline", combo.baseline)]:
                npz[f"{ck}/{label}_train_losses"] = method.fine_tune.train_losses
                npz[f"{ck}/{label}_holdout_losses"] = method.fine_tune.holdout_losses

                for coeff_name, arr in method.jacobian_estimates.items():
                    npz[f"{ck}/{label}/{coeff_name}"] = arr
                for coeff_name, arr in method.jacobian_true.items():
                    npz[f"{ck}/{label}/{coeff_name}_true"] = arr

                if method.pred_errors.size > 0:
                    npz[f"{ck}/{label}/pred_errors"] = method.pred_errors
                if method.weights.size > 0:
                    npz[f"{ck}/{label}/weights"] = method.weights

        # Best combo prediction data
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
        """Serialize to results.json format."""
        tasks_dict: Dict[str, Any] = {}
        for task_name, task_result in self.tasks.items():
            td = task_result.to_json_dict()
            # Fill in fixed_steps from config into each combo's coefficient_recovery
            for key in list(td.keys()):
                if key.startswith("coefficient_recovery_") and isinstance(td[key], dict):
                    td[key]["fixed_steps"] = self.config.fixed_steps
            tasks_dict[task_name] = td

        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": asdict(self.config),
            "tasks": tasks_dict,
        }


# ── Helpers ──────────────────────────────────────────────────────────────


def build_method_result(
    fine_tune_result: FineTuneResult,
    jac_snapshots: list,
    pred_snapshots: list,
    weight_snapshots: list | None = None,
) -> MethodResult:
    """Build MethodResult from raw fine-tune output and Jacobian snapshots."""
    # Coefficient recovery summary
    coefficients: Dict[str, CoefficientSnapshot] = {}
    for name in jac_snapshots[0].estimates:
        coefficients[name] = CoefficientSnapshot(
            true_value=jac_snapshots[0].true_values[name],
            recovered=[jac.recovered(name) for jac in jac_snapshots],
            error_pct=[jac.coeff_error_pct(name) for jac in jac_snapshots],
            mean=[float(np.mean(jac.estimates[name])) for jac in jac_snapshots],
            std=[float(np.std(jac.estimates[name])) for jac in jac_snapshots],
        )

    summary = SnapshotSummary(
        coefficients=coefficients,
        error_pct=[
            float(np.mean([jac.coeff_error_pct(n) for n in jac.true_values]))
            for jac in jac_snapshots
        ],
    )

    # Raw arrays for NPZ
    jac_estimates: Dict[str, np.ndarray] = {}
    jac_true: Dict[str, np.ndarray] = {}
    for name in jac_snapshots[0].estimates:
        jac_estimates[name] = np.stack([jac.estimates[name] for jac in jac_snapshots])
        jac_true[name] = np.array([jac_snapshots[0].true_values[name]])

    return MethodResult(
        fine_tune=fine_tune_result,
        coefficient_recovery=summary,
        jacobian_estimates=jac_estimates,
        jacobian_true=jac_true,
        pred_errors=np.stack(pred_snapshots) if pred_snapshots else np.array([]),
        weights=np.stack(weight_snapshots) if weight_snapshots else np.array([]),
    )


def _flat_to_summary(flat: Dict[str, Any], coeff_names: List[str]) -> SnapshotSummary:
    """Convert flat {name}_true, {name}_recovered, ... dict to SnapshotSummary."""
    coefficients: Dict[str, CoefficientSnapshot] = {}
    for name in coeff_names:
        if f"{name}_true" in flat:
            coefficients[name] = CoefficientSnapshot(
                true_value=flat[f"{name}_true"],
                recovered=flat.get(f"{name}_recovered", []),
                error_pct=flat.get(f"{name}_error_pct", []),
                mean=flat.get(f"{name}_mean", []),
                std=flat.get(f"{name}_std", []),
            )
    return SnapshotSummary(
        coefficients=coefficients,
        error_pct=flat.get("error_pct", []),
    )


def _summary_to_flat(summary: SnapshotSummary) -> Dict[str, Any]:
    """Convert SnapshotSummary to flat {name}_true, {name}_recovered, ... format."""
    d: Dict[str, Any] = {}
    for name, snap in summary.coefficients.items():
        d[f"{name}_true"] = snap.true_value
        d[f"{name}_recovered"] = snap.recovered
        d[f"{name}_error_pct"] = snap.error_pct
        d[f"{name}_mean"] = snap.mean
        d[f"{name}_std"] = snap.std
    d["error_pct"] = summary.error_pct
    return d
