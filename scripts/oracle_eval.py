#!/usr/bin/env python3
"""
Oracle / cheat evaluation: verify auxiliary_losses and extract_coefficients
by feeding the pipeline a *functional oracle* model that implements each
PDE's RHS analytically using the task's true coefficients.

For a perfect oracle, the structural-feature pipeline must satisfy:
  - auxiliary_losses ≈ 0 (only float32 round-off)
  - extract_coefficients returns the true coefficient at every recovery path
  - cross_path_std ≈ 0 (perfect agreement across recovery paths)

The script writes a standard experiment directory at
`data/models/oracle-cheat/{pde}-oracle/{evaluation,training}/...`
that visualize.py reads with no flag changes — same layout as a real eval
run, just with both maml and baseline branches populated by the same oracle.
After the dump, run:

    uv run python scripts/visualize.py --config data/models/oracle-cheat/{pde}-oracle/training/config.yaml

and inspect the figures: every per-coefficient scatter / number-line should
show the recovered points sitting exactly on the truth (or at the truth value
in number-line mode), every histogram should be a delta spike, and MAML and
Baseline panels should be visually identical.

Currently implemented oracles: BR. Other PDEs (FHN, λ-ω, NS, Heat, NLHeat)
will be added as separate oracle classes following the same pattern.

Usage:
    uv run python scripts/oracle_eval.py --pde br
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.evaluation.eval_types import (
    MixerFineTuneOutput,
    assemble_method_result,
)
from src.evaluation.results import (
    ComboResult,
    EvalConfig,
    EvaluationResults,
    TaskResult,
    WorseFlags,
)
from src.training.task_loader import (
    BrusselatorTask,
    MetaLearningDataLoader,
    PDETask,
    TASK_REGISTRY,
)


# ── Oracles ───────────────────────────────────────────────────────────────


class BrusselatorOracle(nn.Module):
    """Functional oracle for Brusselator. Implements the exact PDE RHS.

    BR equations:
        u_t = D_u·∇²u + k1 - (k2+1)·u + u²v
        v_t = D_v·∇²v +      k2·u    - u²v

    Mixer libraries (matching `BrusselatorTask.structural_feature_names`):
        mixer_u: [u, u²v, u_xx+u_yy]   weights (-(k2+1), +1, D_u)  bias k1
        mixer_v: [u, u²v, v_xx+v_yy]   weights (k2,      -1, D_v)  bias 0

    A perfect mixer would learn exactly those weights. The oracle returns
    the linear combination directly using the task's true coefficients, so
    JVPs through the oracle produce constant per-feature partials matching
    the coefficients, and the residual subtraction for k1 returns exactly k1.
    """

    def __init__(self, D_u: float, D_v: float, k1: float, k2: float):
        super().__init__()
        self.D_u = float(D_u)
        self.D_v = float(D_v)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def forward_one(self, mixer_idx: int, features: torch.Tensor) -> torch.Tensor:
        if mixer_idx == 0:
            # mixer_u: [u, u²v, u_xx+u_yy]
            return (
                -(self.k2 + 1.0) * features[:, 0]
                + features[:, 1]
                + self.D_u * features[:, 2]
                + self.k1
            )
        if mixer_idx == 1:
            # mixer_v: [u, u²v, v_xx+v_yy]
            return (
                self.k2 * features[:, 0]
                - features[:, 1]
                + self.D_v * features[:, 2]
            )
        raise ValueError(f"BR has 2 mixers; got mixer_idx={mixer_idx}")


def build_oracle(task: PDETask) -> Any:
    """Dispatch to the per-PDE oracle constructor.

    Return type is `Any` because the per-PDE oracles are duck-typed on
    `forward_one(mixer_idx, features)` — `nn.Module` on its own doesn't
    advertise that method, so a strict annotation would force every call
    site through a cast. Each concrete oracle subclasses `nn.Module` so it
    still works with `.to(device)`.
    """
    if isinstance(task, BrusselatorTask):
        return BrusselatorOracle(
            D_u=task.D_u,
            D_v=task.D_v,
            k1=task.k1,
            k2=task.k2,
        )
    raise NotImplementedError(
        f"Oracle for {type(task).__name__} not implemented yet. "
        f"Add a new *Oracle nn.Module and extend build_oracle()."
    )


# ── Per-task cheat evaluation ─────────────────────────────────────────────


def cheat_evaluate_task(
    task: PDETask,
    k_values: List[int],
    noise_levels: List[float],
    fixed_steps: List[int],
    holdout_size: int,
    seed: int,
    device: str,
    aux_loss_tol: float,
    pct_error_tol: float,
) -> TaskResult:
    """Run the oracle through extract_coefficients + auxiliary_losses for one task.

    Mirrors the structure of `evaluate.py:evaluate_task` but skips the
    fine-tuning loop entirely — the oracle is already perfect, so there's
    nothing to adapt. extract_coefficients is called once per mixer per
    combo (the result is the same at every fixed_step since the model
    doesn't change), then replicated across the fixed_steps positions.

    Both maml and baseline branches use the *same* oracle. The resulting
    figures should show MAML and Baseline panels visually identical — if
    they're not, something is wrong in the merge / serialize / visualize path.
    """
    oracle = build_oracle(task).to(device)

    task_result = TaskResult(
        task_name=task.task_name,
        coefficients=task.true_coefficients,
        ic_type=task.ic_config.get("type", "unknown"),
        n_samples=task.n_samples,
    )

    mixer_names = task.mixer_names
    n_outputs = task.n_outputs
    true_coefficients = task.true_coefficients

    for k_idx, k in enumerate(k_values):
        k_seed = seed + k_idx * 100
        actual_holdout = min(holdout_size, task.n_samples - k)

        for noise_idx, noise in enumerate(noise_levels):
            noise_generator = None
            if noise > 0.0:
                noise_generator = torch.Generator(device=device).manual_seed(
                    k_seed + noise_idx * 1000
                )

            _support, holdout, _support_coords, _holdout_coords = task.get_support_query_split(
                K_shot=k,
                query_size=actual_holdout,
                k_seed=k_seed,
                snapshot_seed=seed,
                noise_level=noise,
                noise_generator=noise_generator,
            )
            holdout_features_list, holdout_targets = holdout

            mixer_outputs: List[MixerFineTuneOutput] = []
            for mixer_idx in range(n_outputs):
                mixer_name = mixer_names[mixer_idx]
                holdout_x_i = holdout_features_list[mixer_idx]

                # ── Coefficient extraction (the thing under test) ──
                extractions = task.extract_coefficients(mixer_idx, oracle, holdout_x_i)

                # ── Auxiliary losses (the other thing under test) ──
                aux = task.auxiliary_losses(
                    mixer_idx, oracle, holdout_x_i, holdout_targets
                )
                for aux_name, aux_val in aux.items():
                    aux_float = float(aux_val.detach().item())
                    assert aux_float < aux_loss_tol, (
                        f"\n  AUX LOSS ASSERTION FAILED:\n"
                        f"  task={task.task_name} mixer={mixer_name} aux={aux_name}\n"
                        f"  value={aux_float:.3e}  tol={aux_loss_tol:.0e}\n"
                        f"  This means the oracle's structural form does not match "
                        f"what auxiliary_losses expects. Either the library is wrong, "
                        f"the oracle math is wrong, or the loss formula has drifted."
                    )

                # ── Verify extraction means against truth ──
                for coeff_name, formulas in extractions.items():
                    true_val = true_coefficients[coeff_name]
                    for formula_tag, ext in formulas.items():
                        ext_val = float(ext.mean.detach().item())
                        if true_val != 0.0:
                            err_pct = abs(ext_val - true_val) / abs(true_val) * 100.0
                        else:
                            err_pct = abs(ext_val - true_val) * 100.0
                        assert err_pct < pct_error_tol, (
                            f"\n  EXTRACTION ASSERTION FAILED:\n"
                            f"  task={task.task_name} mixer={mixer_name} coeff={coeff_name} "
                            f"path={formula_tag}\n"
                            f"  recovered={ext_val:.6f}  true={true_val:.6f}  "
                            f"err={err_pct:.4f}% (tol={pct_error_tol:.0e}%)\n"
                            f"  Either extract_coefficients has a bug or the oracle "
                            f"implementation does not actually match the PDE form."
                        )

                # Replicate the extraction across all fixed_steps. The oracle
                # is static so every snapshot has the same value — this lets
                # downstream visualize.py (which expects per_step_extractions
                # populated at every fixed_step) consume the result without
                # special-casing.
                per_step_ext = {step: extractions for step in fixed_steps}

                # Per-step prediction errors on holdout. Oracle pred should
                # exactly match the analytical RHS, which equals the target
                # (which IS the analytical RHS — see fourier_eval). So
                # pred_errors should be ~float32 zero.
                with torch.no_grad():
                    pred = oracle.forward_one(mixer_idx, holdout_x_i)
                    target_col = holdout_targets[:, mixer_idx]
                    pred_err = (pred - target_col).abs().detach().cpu().numpy()
                pred_errors_per_step = {step: pred_err for step in fixed_steps}

                # Synthetic losses. Use the actual nMSE on the holdout so the
                # figures show the oracle's true loss (~float32 floor).
                with torch.no_grad():
                    target_sq = (target_col ** 2).mean().clamp(min=1e-12)
                    loss_value = float(
                        ((pred - target_col) ** 2).mean().item()
                        / target_sq.item()
                    )
                losses = np.full(len(fixed_steps), loss_value, dtype=np.float64)

                mixer_outputs.append(
                    MixerFineTuneOutput(
                        mixer_name=mixer_name,
                        train_losses=losses,
                        holdout_losses=losses,
                        per_step_extractions=per_step_ext,
                        pred_errors_per_step=pred_errors_per_step,
                    )
                )

            # Same oracle on both branches — maml and baseline are identical.
            # The figures should show identical MAML and Baseline panels;
            # any visible difference indicates a bug in serialization or
            # the visualize.py per-method dispatch.
            maml_method = assemble_method_result(
                mixer_outputs=mixer_outputs,
                true_coefficients=true_coefficients,
                fixed_steps=fixed_steps,
            )
            baseline_method = assemble_method_result(
                mixer_outputs=mixer_outputs,
                true_coefficients=true_coefficients,
                fixed_steps=fixed_steps,
            )

            task_result.combos.append(
                ComboResult(
                    k=k,
                    noise=noise,
                    maml=maml_method,
                    baseline=baseline_method,
                    worse=WorseFlags(),
                )
            )

    return task_result


# ── Stub training/config.yaml so visualize.py can find the experiment ───


def _write_stub_config(
    exp_dir: Path,
    pde: str,
    out_name: str,
    base_dir: str,
    meta_test_dir: str,
    k_values: List[int],
    noise_levels: List[float],
    fixed_steps: List[int],
    holdout_size: int,
) -> Path:
    """Write a minimal ExperimentConfig YAML for the oracle dir.

    visualize.py reads `experiment.name`, `output.base_dir`,
    `experiment.pde_type`, `data.meta_train_dir`, `evaluation.*`, and
    `visualization.*` — only those need real values. Everything else is
    set to schema-valid placeholders.
    """
    cfg_dict: Dict = {
        "experiment": {
            "name": out_name,
            "pde_type": pde,
            "seed": 42,
            "device": "cpu",
        },
        "output": {"base_dir": base_dir},
        "data": {
            "meta_train_dir": meta_test_dir,
            "meta_val_dir": meta_test_dir,
            "meta_test_dir": meta_test_dir,
        },
        "training": {
            "inner_lr": 0.0,
            "outer_lr": 0.0,
            "adam_betas": [0.9, 0.999],
            "inner_steps": fixed_steps[-1],
            "meta_batch_size": 1,
            "k_shot": k_values[0],
            "query_size": holdout_size,
            "epochs": 1,
            "max_iterations": 1,
            "patience": 0,
            "checkpoint_interval": 1,
            "log_interval": 1,
            "hidden_dims": [350, 350],
            "activation": "silu",
            "input_dim": 2,
            "output_dim": 1,
            "input_bypass": False,
            "first_order": False,
            "msl_enabled": False,
            "da_enabled": False,
            "da_threshold": 5000,
            "lslr_enabled": False,
            "warmup_iterations": 0,
            "use_scheduler": False,
            "scheduler_type": "polynomial",
            "T_0": 200,
            "T_mult": 2,
            "min_lr": 1.0e-07,
            "poly_power": 3.0,
            "plateau_factor": 0.8,
            "plateau_patience": 70,
            "plateau_cooldown": 40,
            "plateau_threshold": 0.001,
            "plateau_window": 20,
            "loss_function": "normalized_mse",
            "max_grad_norm": 0.0,
            "aux_losses_enabled": True,
            "imaml": {
                "enabled": True,
                "lam": 0.005,
                "lam_lr": 0.0,
                "lam_min": 0.0,
                "cg_steps": 10,
                "cg_damping": 1.0,
                "inner_optimizer": "lbfgs",
                "outer_optimizer": "adam",
                "outer_lbfgs_after": 0,
                "anil": True,
                "proximal_every_step": False,
            },
        },
        "evaluation": {
            "k_values": k_values,
            "noise_levels": noise_levels,
            "fine_tune_lr": 0.0,
            "max_steps": fixed_steps[-1],
            "deriv_threshold": 0.0005,
            "fixed_steps": fixed_steps,
            "holdout_size": holdout_size,
            "log_weights": False,
        },
        "visualization": {
            "dpi": 150,
            "only": "",
            "compare_experiments": [],
            "exclude_suffixes_append": [],
            "exclude_max_iteration": 20,
        },
    }
    training_dir = exp_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = training_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    return cfg_path


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Oracle (cheat) evaluation — verify aux losses and coefficient extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pde",
        type=str,
        default="br",
        choices=["br"],
        help="PDE to test (currently: br).",
    )
    parser.add_argument(
        "--meta-test-dir",
        type=Path,
        default=Path("data/datasets/br_test-2"),
        help="Directory containing test task NPZs.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/models/oracle-cheat"),
        help="Output base directory for the cheat experiment.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Experiment name (default: {pde}-oracle).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu — oracle is cheap).",
    )
    parser.add_argument("--k", type=int, default=1000, help="Support set size (single).")
    parser.add_argument("--holdout", type=int, default=10000, help="Holdout size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--aux-tol", type=float, default=1e-5,
        help="Auxiliary loss tolerance (asserted per coeff per task).",
    )
    parser.add_argument(
        "--pct-tol", type=float, default=1e-2,
        help="Coefficient pct_error tolerance (asserted per path per task).",
    )
    args = parser.parse_args()

    pde = args.pde
    out_name = args.out_name or f"{pde}-oracle"

    k_values = [args.k]
    noise_levels = [0.0]
    fixed_steps = [0, 16]
    holdout_size = args.holdout

    print("=" * 60)
    print(f"Oracle evaluation: {pde}")
    print("=" * 60)
    print(f"  meta_test_dir: {args.meta_test_dir}")
    print(f"  base_dir:      {args.base_dir}")
    print(f"  out_name:      {out_name}")
    print(f"  device:        {args.device}")
    print(f"  K:             {k_values}")
    print(f"  noise:         {noise_levels}")
    print(f"  fixed_steps:   {fixed_steps}")
    print(f"  holdout_size:  {holdout_size}")
    print(f"  aux_loss_tol:  {args.aux_tol:.0e}")
    print(f"  pct_error_tol: {args.pct_tol:.0e} %")
    print()

    task_class = TASK_REGISTRY[pde]
    test_loader = MetaLearningDataLoader(
        args.meta_test_dir,
        task_class=task_class,
        task_pattern="*_fourier.npz",
        device=args.device,
    )
    print(f"Loaded {len(test_loader)} test tasks")
    print()

    tasks_dict: Dict[str, TaskResult] = {}
    for task_idx, task in enumerate(test_loader.tasks):
        print(f"[{task_idx + 1}/{len(test_loader)}] {task.task_name} ", end="", flush=True)
        result = cheat_evaluate_task(
            task=task,
            k_values=k_values,
            noise_levels=noise_levels,
            fixed_steps=fixed_steps,
            holdout_size=holdout_size,
            seed=args.seed + task_idx * 10000,
            device=args.device,
            aux_loss_tol=args.aux_tol,
            pct_error_tol=args.pct_tol,
        )
        tasks_dict[task.task_name] = result
        print("OK")

    print()
    print("All assertions passed.")
    print()

    # Build EvaluationResults and serialize.
    config = EvalConfig(
        k_values=k_values,
        noise_levels=noise_levels,
        fine_tune_lr=0.0,
        max_steps=fixed_steps[-1],
        threshold=0.0005,
        fixed_steps=fixed_steps,
        holdout_size=holdout_size,
        pde_type=pde,
    )
    eval_results = EvaluationResults(
        experiment_name=out_name,
        timestamp=datetime.now().isoformat(),
        config=config,
        tasks=tasks_dict,
    )

    exp_dir = args.base_dir / out_name
    eval_dir = exp_dir / "evaluation"
    samples_dir = eval_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    results_dict = eval_results.to_json_dict()
    with open(eval_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=lambda o: None if callable(o) else str(o))

    # samples NPZs
    for task_name, task_result in tasks_dict.items():
        npz = task_result.to_npz_dict()
        if npz:
            np.savez_compressed(samples_dir / f"{task_name}.npz", **npz)

    # Stub training/config.yaml so visualize.py can read the experiment.
    cfg_path = _write_stub_config(
        exp_dir=exp_dir,
        pde=pde,
        out_name=out_name,
        base_dir=str(args.base_dir),
        meta_test_dir=str(args.meta_test_dir),
        k_values=k_values,
        noise_levels=noise_levels,
        fixed_steps=fixed_steps,
        holdout_size=holdout_size,
    )

    # DONE sentinel for symmetry with evaluate.py.
    (eval_dir / "DONE").touch()

    print(f"Output:      {eval_dir}")
    print(f"Stub config: {cfg_path}")
    print()
    print("Visualize:")
    print(f"    uv run python scripts/visualize.py --config {cfg_path}")


if __name__ == "__main__":
    main()
