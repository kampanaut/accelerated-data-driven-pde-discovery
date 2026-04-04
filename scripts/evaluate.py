#!/usr/bin/env python3
"""
Evaluation script for MAML vs baseline comparison.

This script:
1. Loads meta-learned θ* and initial θ₀ from checkpoints
2. Loads meta-test tasks
3. For each (task, K, noise) combination:
   - Samples K points ONCE (fair comparison)
   - Fine-tunes from θ* (MAML) and θ₀ (baseline)
   - Records loss trajectories, coefficient estimates, prediction errors
4. Saves results.json (metadata) and samples/*.npz (arrays)

Usage:
    python scripts/evaluate.py --config configs/experiment.yaml
"""

import io
import sys
import copy
import json
import shutil
import argparse


from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Callable, Optional


from numpy.typing import NDArray
import torch
torch.set_num_threads(9)
torch.set_num_interop_threads(4)
import torch.nn.functional as F
import numpy as np

from src.config import ExperimentConfig
from src.networks.pde_operator_network import NetworkConfig, PDEOperatorNetwork
from src.training.task_loader import MetaLearningDataLoader, PDETask, TASK_REGISTRY
from src.training.maml import MeTALModule, LSLRSchedule
from src.training.spectral_loss import compute_spectral_loss
from src.evaluation.jacobian import analyze_jacobian, JacobianResults
from src.evaluation.metrics import compress_step_ranges
from src.evaluation.results import (
    FineTuneResult, MethodResult, ComboResult, TaskResult,
    WorseFlags, BestComboData, build_method_result,
)


class _TeeStream:
    """Write to both stdout and a StringIO buffer."""

    def __init__(self, original: object):  # type: ignore[arg-type]
        self.original = original
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.original.write(text)  # type: ignore[union-attr]
        self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()  # type: ignore[union-attr]

    def getvalue(self) -> str:
        return self.buffer.getvalue()



def load_model_from_checkpoint(
    checkpoint_path: Path, device: str, net_config: NetworkConfig
) -> PDEOperatorNetwork:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model to
        net_config: Network architecture config

    Returns:
        Loaded PDEOperatorNetwork
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PDEOperatorNetwork(net_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def fine_tune(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    lr: float,
    max_steps: int,
    holdout_features: torch.Tensor,
    holdout_targets: torch.Tensor,
    fixed_steps: Optional[List[int]] = None,
    on_step: Optional[Callable[[torch.nn.Module, int], None]] = None,
    metal: Optional[MeTALModule] = None,
    lslr: Optional["LSLRSchedule"] = None,
    loss_type: str = "normalized_mse",
    coords: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    Lx: float = 0.0,
    Ly: float = 0.0,
    spectral_mode_size: int = 0,
    max_grad_norm: float = 0.0,
    proximal_lam: float = 0.0,
    proximal_theta: Optional[torch.Tensor] = None,
    inner_optimizer: str = "sgd",
) -> FineTuneResult:
    """
    Fine-tune model and return train/holdout loss at each step.

    Args:
        model: Model to fine-tune (will be modified in-place)
        features: Training input features tensor (N, 10) on device
        targets: Training target outputs tensor (N, 2) on device
        lr: Learning rate
        max_steps: Number of gradient steps
        holdout_features: Holdout input features tensor for generalization eval
        holdout_targets: Holdout targets tensor for generalization eval
        fixed_steps: Steps at which to call on_step (0 = pre-training, e.g. [0, 1, 10, 50])
        on_step: Callback called at each fixed_step with (model, step_number)
        metal: Frozen MeTALModule (None = standard loss)
        loss_type: Loss function — 'mse', 'normalized_mse', or 'mae'

    Returns:
        Dict with 'train_losses' and 'holdout_losses'
    """
    model.train()

    # Inner optimizer: SGD (with optional LSLR) or L-BFGS
    use_lbfgs = (inner_optimizer == "lbfgs")

    if use_lbfgs:
        param_names = None
        lr_schedule = None
        opt = torch.optim.LBFGS(
            model.parameters(), lr=1.0,
            max_iter=1, line_search_fn="strong_wolfe",
        )
    elif lslr is not None:
        param_names = [n for n, _ in model.named_parameters()]
        opt = torch.optim.SGD(
            [{"params": [p], "lr": lr} for p in model.parameters()]
        )
        lr_schedule = {
            name: [
                lslr.lr_dict[name.replace(".", "-")][s].item()
                for s in range(lslr.n_steps)
            ]
            for name in param_names
        }
    else:
        param_names = None
        lr_schedule = None
        opt = torch.optim.SGD(model.parameters(), lr=lr)

    # Proximal term for iMAML evaluation
    def _proximal_loss() -> torch.Tensor:
        if proximal_lam > 0 and proximal_theta is not None:
            phi = torch.cat([p.view(-1) for p in model.parameters()])
            return 0.5 * proximal_lam * (phi - proximal_theta).pow(2).sum()
        return torch.tensor(0.0, device=features.device)

    if metal is not None:
        metal.eval()

    # Resolve pointwise loss once — no string dispatch per call
    if loss_type == "mse":
        _pw: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            lambda p, t: F.mse_loss(p, t)
        )
    elif loss_type == "normalized_mse":
        _pw = lambda p, t: F.mse_loss(p, t) / (t**2).mean()
    elif loss_type == "mae":
        _pw = lambda p, t: F.l1_loss(p, t)
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Use 'mse', 'normalized_mse', or 'mae'."
        )

    # Metric: always clean pointwise loss (for recording — comparable across experiments)
    metric_fn = _pw

    # Training cost: pointwise + optional spectral (drives gradient updates only)
    use_spectral = coords is not None and spectral_mode_size > 0
    if use_spectral:
        assert coords is not None
        _x_pts, _y_pts = coords

        def cost_fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            pw = _pw(p, t)
            spec = compute_spectral_loss(
                p, t, _x_pts, _y_pts, Lx, Ly, spectral_mode_size
            )
            return pw + spec
    else:
        cost_fn = _pw

    x = features
    y = targets
    x_holdout = holdout_features
    y_holdout = holdout_targets

    fixed_steps_set = set(fixed_steps) if fixed_steps else set()

    train_losses: List[float] = []
    holdout_losses: List[float] = []

    # Step 0: record losses + snapshot before any gradient update
    with torch.no_grad():
        pred_0 = model(x)
        train_losses.append(metric_fn(pred_0, y).item())
        pred_h0 = model(x_holdout)
        holdout_losses.append(metric_fn(pred_h0, y_holdout).item())

    if on_step is not None and 0 in fixed_steps_set:
        on_step(model, 0)
        model.train()

    def _holdout_and_callback(step_idx: int) -> None:
        """Record holdout loss and fire callback after a step."""
        with torch.no_grad():
            pred_holdout = model(x_holdout)
            holdout_loss = metric_fn(pred_holdout, y_holdout)
            holdout_losses.append(holdout_loss.item())

        step_num = step_idx + 1
        if on_step is not None and step_num in fixed_steps_set:
            on_step(model, step_num)
            model.train()

    def _step_sgd(step_idx: int, grad_loss: torch.Tensor) -> None:
        """SGD step: backward, clip, LSLR override, optimize."""
        total_loss = grad_loss + _proximal_loss()
        opt.zero_grad()
        total_loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if lr_schedule is not None and param_names is not None:
            if step_idx < lslr.n_steps:  # type: ignore[union-attr]
                for pg, name in zip(opt.param_groups, param_names):
                    pg["lr"] = lr_schedule[name][step_idx]
            else:
                for pg in opt.param_groups:
                    pg["lr"] = lr
        opt.step()
        _holdout_and_callback(step_idx)

    def _step_lbfgs(step_idx: int, grad_loss: torch.Tensor) -> None:
        """L-BFGS step: closure with cost + proximal."""
        def closure():
            opt.zero_grad()
            p = model(x)
            gl = cost_fn(p, y) + _proximal_loss()
            gl.backward()
            return gl

        opt.step(closure)
        _holdout_and_callback(step_idx)

    def _step_lbfgs_metal(step_idx: int, grad_loss: torch.Tensor) -> None:
        """L-BFGS step: closure with cost + proximal + MeTAL."""
        def closure():
            opt.zero_grad()
            p = model(x)
            gl = cost_fn(p, y) + _proximal_loss()
            ms = metal.support_step(step_idx, model, p, y, cost_fn)  # type: ignore[union-attr]
            gl = gl + ms
            gl.backward()
            return gl

        opt.step(closure)
        _holdout_and_callback(step_idx)

    if use_lbfgs:
        _step = _step_lbfgs_metal if metal is not None else _step_lbfgs
    else:
        _step = _step_sgd

    # Phase 1: MeTAL-shaped inner steps (if MeTAL provided)
    metal_steps = metal.n_steps if metal is not None else 0
    for i in range(min(metal_steps, max_steps)):
        pred = model(x)
        train_losses.append(metric_fn(pred, y).item())

        grad_loss = cost_fn(pred, y)
        meta_support = metal.support_step(i, model, pred, y, cost_fn)  # type: ignore[union-attr]
        _step(i, grad_loss + meta_support)

    # Phase 2: Standard loss for remaining steps
    for i in range(metal_steps, max_steps):
        pred = model(x)
        train_losses.append(metric_fn(pred, y).item())

        grad_loss = cost_fn(pred, y)
        _step(i, grad_loss)

    return FineTuneResult(
        train_losses=np.array(train_losses),
        holdout_losses=np.array(holdout_losses),
    )


def evaluate_task(
    task: PDETask,
    theta_star: PDEOperatorNetwork,
    theta_0: PDEOperatorNetwork,
    k_values: List[int],
    noise_levels: List[float],
    fine_tune_lr: float,
    max_steps: int,
    device: str,
    seed: int,
    fixed_steps: List[int],
    holdout_size: int = 1000,
    metal: Optional[MeTALModule] = None,
    lslr: Optional["LSLRSchedule"] = None,
    loss_type: str = "normalized_mse",
    spectral_mode_size: int = 0,
    max_grad_norm: float = 0.0,
    log_weights: bool = False,
    zero_non_rhs_features: bool = False,
    proximal_lam: float = 0.0,
    proximal_theta: Optional[torch.Tensor] = None,
    inner_optimizer: str = "sgd",
) -> TaskResult:
    """
    Evaluate one task across all (K, noise) combinations.

    Extracts Jacobian and pred_errors at every fixed_step during fine-tuning.

    Returns:
        TaskResult with all combo results, NPZ data, and best-combo predictions.
    """
    specs = task.coefficient_specs

    task_result = TaskResult(
        task_name=task.task_name,
        coefficients=task.diffusion_coeffs,
        coefficient_specs=list(specs),
        ic_type=task.ic_config.get("type", "unknown"),
        n_samples=task.n_samples,
    )

    for k_idx, k in enumerate(k_values):
        # Evaluate Fourier ONCE per (task, K) — same collocation points for all noise levels
        k_seed = seed + k_idx * 100
        available_for_holdout = task.n_samples - k
        actual_holdout = min(holdout_size, available_for_holdout)

        support_clean, holdout_clean, support_coords, holdout_coords = task.get_support_query_split(
            K_shot=k,
            query_size=actual_holdout,
            k_seed=k_seed,
            snapshot_seed=seed,
        )

        for noise_idx, noise in enumerate(noise_levels):
            print(f"    K={k:4d}, noise={noise:.0%}...", end=" ", flush=True)

            try:
                # Inject noise (or use clean data directly)
                if noise == 0.0:
                    features, targets = support_clean
                    holdout_features, holdout_targets = holdout_clean
                else:
                    noise_gen = torch.Generator(device=device)
                    noise_gen.manual_seed(k_seed + noise_idx)
                    feat_s, tgt_s = support_clean[0].clone(), support_clean[1].clone()
                    feat_h, tgt_h = holdout_clean[0].clone(), holdout_clean[1].clone()
                    feat_s, tgt_s = task.inject_noise(
                        feat_s, tgt_s, noise, generator=noise_gen
                    )
                    feat_h, tgt_h = task.inject_noise(
                        feat_h, tgt_h, noise, generator=noise_gen
                    )
                    features, targets = feat_s, tgt_s
                    holdout_features, holdout_targets = feat_h, tgt_h

                # Zero non-RHS features (after noise injection)
                if zero_non_rhs_features:
                    features = task.zero_non_rhs_features(features)
                    holdout_features = task.zero_non_rhs_features(holdout_features)

                # --- Build callback for intermediate Jacobian extraction ---
                maml_jac_snapshots: List[JacobianResults] = []
                maml_pred_snapshots: List[NDArray[np.integer[Any]]] = []
                maml_weight_snapshots: List[np.ndarray] = []
                baseline_jac_snapshots: List[JacobianResults] = []
                baseline_pred_snapshots: List[NDArray[np.integer[Any]]] = []
                baseline_weight_snapshots: List[np.ndarray] = []

                _WEIGHT_LABELS_5 = ["u", "u_x", "u_y", "u_xx", "u_yy"]
                _WEIGHT_LABELS_10 = ["u", "v", "u_x", "u_y", "u_xx", "u_yy",
                                     "v_x", "v_y", "v_xx", "v_yy"]

                def _make_on_step(
                    jac_list: List[JacobianResults],
                    pred_list: List[np.ndarray],
                    h_feat: torch.Tensor,
                    h_tgt: torch.Tensor,
                    weight_list: Optional[List[np.ndarray]] = None,
                    method_label: str = "",
                ) -> Callable[[torch.nn.Module, int], None]:
                    _printed_header = [False]  # mutable flag for newline before first weight log

                    def on_step(model: torch.nn.Module, step: int) -> None:
                        jac = analyze_jacobian(model, h_feat, specs, device=device)
                        jac_list.append(jac)
                        with torch.no_grad():
                            pred = model(h_feat)
                            pred_errors = (pred - h_tgt).abs().cpu().numpy()
                        pred_list.append(pred_errors)
                        if weight_list is not None:
                            if not _printed_header[0]:
                                print()  # newline after "K=..., noise=...%" line
                                _printed_header[0] = True
                            flat = torch.cat([
                                p.detach().cpu().flatten() for p in model.parameters()
                            ]).numpy()
                            weight_list.append(flat)
                            n = len(flat)
                            labels = (
                                _WEIGHT_LABELS_5 if n == 5
                                else _WEIGHT_LABELS_10 if n == 10
                                else [f"w{i}" for i in range(n)]
                            )
                            parts = [f"{l}={v:+.4f}" for l, v in zip(labels, flat)]
                            print(f"      [{method_label:>4s}] step {step:4d}: {'  '.join(parts)}")

                    return on_step

                # Fine-tune from θ* (MAML) — uses MeTAL loss if networks provided
                maml_model = copy.deepcopy(theta_star)
                maml_result = fine_tune(
                    maml_model,
                    features,
                    targets,
                    fine_tune_lr,
                    max_steps,
                    holdout_features=holdout_features,
                    holdout_targets=holdout_targets,
                    fixed_steps=fixed_steps,
                    on_step=_make_on_step(
                        maml_jac_snapshots,
                        maml_pred_snapshots,
                        holdout_features,
                        holdout_targets,
                        maml_weight_snapshots if log_weights else None,
                        method_label="MAML",
                    ),
                    metal=metal,
                    lslr=lslr,
                    loss_type=loss_type,
                    coords=support_coords,
                    Lx=task.Lx,
                    Ly=task.Ly,
                    spectral_mode_size=spectral_mode_size,
                    max_grad_norm=max_grad_norm,
                    proximal_lam=proximal_lam,
                    proximal_theta=proximal_theta,
                    inner_optimizer=eval_inner_optimizer,
                )

                # Fine-tune from θ₀ (baseline) — same loss, no MeTAL, no LSLR
                baseline_model = copy.deepcopy(theta_0)
                baseline_result = fine_tune(
                    baseline_model,
                    features,
                    targets,
                    fine_tune_lr,
                    max_steps,
                    holdout_features=holdout_features,
                    holdout_targets=holdout_targets,
                    fixed_steps=fixed_steps,
                    on_step=_make_on_step(
                        baseline_jac_snapshots,
                        baseline_pred_snapshots,
                        holdout_features,
                        holdout_targets,
                        baseline_weight_snapshots if log_weights else None,
                        method_label="BL",
                    ),
                    loss_type=loss_type,
                    coords=support_coords,
                    Lx=task.Lx,
                    Ly=task.Ly,
                    spectral_mode_size=spectral_mode_size,
                    max_grad_norm=max_grad_norm,
                )

                # Build MethodResult for each method
                maml_method = build_method_result(
                    maml_result, maml_jac_snapshots, maml_pred_snapshots,
                    maml_weight_snapshots if log_weights else None,
                )
                baseline_method = build_method_result(
                    baseline_result, baseline_jac_snapshots, baseline_pred_snapshots,
                    baseline_weight_snapshots if log_weights else None,
                )

                # Per-step worse comparison
                loss_worse_steps: List[int] = []
                coeff_worse_steps: Dict[str, List[int]] = {
                    n: [] for n in maml_jac_snapshots[0].true_values.keys()
                }

                for si, step in enumerate(fixed_steps):
                    maml_h = maml_result.holdout_losses[step]
                    baseline_h = baseline_result.holdout_losses[step]
                    if maml_h > baseline_h:
                        loss_worse_steps.append(step)

                    for n in maml_jac_snapshots[si].true_values:
                        if maml_jac_snapshots[si].coeff_error_pct(
                            n
                        ) > baseline_jac_snapshots[si].coeff_error_pct(n):
                            coeff_worse_steps[n].append(step)

                combo = ComboResult(
                    k=k,
                    noise=noise,
                    maml=maml_method,
                    baseline=baseline_method,
                    worse=WorseFlags(
                        loss_steps=loss_worse_steps,
                        coeff_steps=coeff_worse_steps,
                    ),
                )

                # Accumulate task-level flags (union across combos)
                for step in loss_worse_steps:
                    if step not in task_result.worse.loss_steps:
                        task_result.worse.loss_steps.append(step)
                for n, steps in coeff_worse_steps.items():
                    if n not in task_result.worse.coeff_steps:
                        task_result.worse.coeff_steps[n] = []
                    for step in steps:
                        if step not in task_result.worse.coeff_steps[n]:
                            task_result.worse.coeff_steps[n].append(step)

                # Print summary
                maml_train_final = maml_result.train_losses[-1]
                maml_holdout_final = maml_result.holdout_losses[-1]
                baseline_train_final = baseline_result.train_losses[-1]
                baseline_holdout_final = baseline_result.holdout_losses[-1]

                flags = []
                if loss_worse_steps:
                    flags.append(
                        f"loss@[{compress_step_ranges(loss_worse_steps, fixed_steps)}]"
                    )
                for n, steps in coeff_worse_steps.items():
                    if steps:
                        flags.append(
                            f"{n}@[{compress_step_ranges(steps, fixed_steps)}]"
                        )

                flag_str = ""
                if len(flags) > 0:
                    flag_str = f" !!! [{','.join(flags)}]"

                print(
                    f"MAML(train={maml_train_final:.2e}, holdout={maml_holdout_final:.2e}) "
                    f"BL(train={baseline_train_final:.2e}, holdout={baseline_holdout_final:.2e}){flag_str}"
                )

                task_result.combos.append(combo)

            except Exception as e:
                import traceback
                print(f"\n{'=' * 60}")
                print(f"FATAL: {type(e).__name__} during combo K={k}, noise={noise:.0%}:")
                print(f"{'=' * 60}")
                traceback.print_exc()
                print(f"{'=' * 60}")
                print("Fix the issue before running evaluation again.")
                sys.exit(1)

    # ─── Best-combo prediction capture ─────────────────────────────────────
    # Re-fine-tune the best combo and record model predictions at each step
    # for spatial visualization of how the model adapts.
    if task_result.combos:
        # Pick combo whose best snapshot has lowest mean coefficient error
        best = min(task_result.combos, key=lambda c: min(c.maml.coefficient_recovery.error_pct))
        best_k = best.k
        best_k_idx = k_values.index(best_k)
        best_noise = best.noise
        best_noise_idx = noise_levels.index(best_noise)
        best_k_seed = seed + best_k_idx * 100

        print(f"    Best combo: {best.combo_key} (min error={min(best.maml.coefficient_recovery.error_pct):.1f}%)")  # type: ignore[union-attr]

        # Re-create the same split (deterministic seed)
        available_for_holdout = task.n_samples - best_k
        actual_holdout = min(holdout_size, available_for_holdout)
        s_clean, h_clean, s_coords, h_coords = task.get_support_query_split(
            K_shot=best_k,
            query_size=actual_holdout,
            k_seed=best_k_seed,
            snapshot_seed=seed
        )

        # Apply noise if needed
        if best_noise == 0.0:
            bc_features, bc_targets = s_clean
            bc_h_features, bc_h_targets = h_clean
        else:
            noise_gen = torch.Generator(device=device)
            noise_gen.manual_seed(best_k_seed + best_noise_idx)
            bc_features, bc_targets = task.inject_noise(
                s_clean[0].clone(), s_clean[1].clone(), best_noise, generator=noise_gen
            )
            bc_h_features, bc_h_targets = task.inject_noise(
                h_clean[0].clone(), h_clean[1].clone(), best_noise, generator=noise_gen
            )

        # Zero non-RHS features (after noise injection)
        if zero_non_rhs_features:
            bc_features = task.zero_non_rhs_features(bc_features)
            bc_h_features = task.zero_non_rhs_features(bc_h_features)

        bc_model = copy.deepcopy(theta_star)
        pred_snapshots: List[NDArray[np.floating[Any]]] = []

        def _pred_on_step(model: torch.nn.Module, _: int) -> None:
            with torch.no_grad():
                pred_snapshots.append(model(bc_h_features).cpu().numpy())

        fine_tune(
            bc_model,
            bc_features,
            bc_targets,
            fine_tune_lr,
            max_steps,
            holdout_features=bc_h_features,
            holdout_targets=bc_h_targets,
            fixed_steps=fixed_steps,
            on_step=_pred_on_step,
            metal=metal,
            lslr=lslr,
            loss_type=loss_type,
            coords=s_coords,
            Lx=task.Lx,
            Ly=task.Ly,
            spectral_mode_size=spectral_mode_size,
            max_grad_norm=max_grad_norm,
            proximal_lam=proximal_lam,
            proximal_theta=proximal_theta,
            inner_optimizer=inner_optimizer,
        )

        step_errors = best.maml.coefficient_recovery.error_pct  # type: ignore[union-attr]

        task_result.best_combo = BestComboData(
            combo_key=best.combo_key,
            predictions=np.stack(pred_snapshots),
            true_targets=bc_h_targets.cpu().numpy(),
            x_pts=h_coords[0].cpu().numpy(),
            y_pts=h_coords[1].cpu().numpy(),
            steps=np.array(fixed_steps),
            coeff_error=np.array(step_errors),
        )

    return task_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MAML vs baseline on meta-test tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to θ* checkpoint (default: final_model.pt in experiment dir)",
    )
    args = parser.parse_args()

    # =========================================================================
    # Load configuration
    # =========================================================================
    print("=" * 60)
    print("MAML Evaluation: θ* vs θ₀ Comparison")
    print("=" * 60)
    print()

    cfg = ExperimentConfig.from_yaml(args.config)
    exp_name = cfg.experiment.name
    base_dir = Path(cfg.output.base_dir)
    exp_dir = base_dir / exp_name

    if not exp_dir.exists():
        # Scan for suffixed directories (ENDNAN → usable, ISNAN → skip)
        candidates = sorted(base_dir.glob(f"{exp_name}-*"))
        endnan = [d for d in candidates if d.name.startswith(f"{exp_name}-ENDNAN")]
        isnan = [d for d in candidates if d.name.startswith(f"{exp_name}-ISNAN")]

        if endnan:
            exp_dir = endnan[0]
            print(f"Using flagged directory: {exp_dir.name}")
        elif isnan:
            print(f"Skipping {exp_name}: flagged as {isnan[0].name}")
            sys.exit(0)
        else:
            print(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)

    # Guard: DONE sentinel means training completed — never overwrite
    # Create samples directory
    eval_dir = exp_dir / "evaluation"
    samples_dir = eval_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    done_path = eval_dir / "DONE"
    if done_path.exists():
        print(f"SKIP: {eval_dir.name} already evaluated (DONE sentinel exists). Delete DONE to overwrite.")
        sys.exit(0)

    # Tee stdout to capture all print output for log file
    _tee = _TeeStream(sys.stdout)
    sys.stdout = _tee  # type: ignore[assignment]

    print(f"Experiment: {exp_name}")
    print(f"Experiment directory: {exp_dir}")
    print()

    # =========================================================================
    # Setup
    # =========================================================================
    seed = cfg.experiment.seed
    device = cfg.experiment.device
    pde_type = cfg.experiment.pde_type

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"PDE type: {pde_type}")
    print()

    # =========================================================================
    # Load checkpoints
    # =========================================================================
    print("-" * 60)
    print("Loading checkpoints...")
    print("-" * 60)

    # θ* (meta-learned) — fall back to latest_model.pt for ENDNAN experiments
    if args.checkpoint is not None:
        theta_star_path = args.checkpoint
    else:
        final_path = exp_dir / "checkpoints" / "final_model.pt"
        best_path = exp_dir / "checkpoints" / "best_model.pt"  # legacy name
        latest_path = exp_dir / "checkpoints" / "latest_model.pt"
        if final_path.exists():
            theta_star_path = final_path
        elif best_path.exists():
            theta_star_path = best_path
            print(f"Using legacy best_model.pt")
        elif latest_path.exists():
            theta_star_path = latest_path
            print(f"final_model.pt not found — using latest_model.pt")
        else:
            print(f"No checkpoint found in {exp_dir / 'checkpoints'}")
            sys.exit(1)

    # θ₀ (initial)
    theta_0_path = exp_dir / "checkpoints" / "initial_model.pt"
    if not theta_0_path.exists():
        raise FileNotFoundError(f"θ₀ checkpoint not found: {theta_0_path}")

    # Get model architecture from training config
    net_config = cfg.to_network_config()

    theta_star = load_model_from_checkpoint(theta_star_path, device, net_config)
    theta_0 = load_model_from_checkpoint(theta_0_path, device, net_config)

    print(f"θ* loaded from: {theta_star_path}")
    print(f"θ₀ loaded from: {theta_0_path}")

    # Load MeTAL module if it exists (frozen for evaluation)
    metal_module: Optional[MeTALModule] = None

    checkpoint_dir = theta_star_path.parent
    metal_state_path = checkpoint_dir / "metal_state.pt"

    if metal_state_path.exists():
        metal_state = torch.load(metal_state_path, map_location=device, weights_only=True)

        # NaN guard: poisoned MeTAL state contaminates all MAML evaluations
        has_nan = any(
            torch.isnan(v).any().item()
            for v in metal_state.values()
            if isinstance(v, torch.Tensor)
        )
        if has_nan:
            print(f"FATAL: MeTAL state contains NaN: {metal_state_path}")
            print(f"Cleaning up evaluation directory...")
            eval_dir = exp_dir / "evaluation"
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
                print(f"Deleted: {eval_dir}")
            sys.exit(1)

        n_base_params = sum(1 for _ in theta_star.parameters())
        output_dim = net_config.output_dim
        inner_steps = cfg.training.inner_steps
        metal_hidden_dim = cfg.training.metal.hidden_dim

        metal_module = MeTALModule(
            n_steps=inner_steps,
            n_base_params=n_base_params,
            output_dim=output_dim,
            hidden_dim=metal_hidden_dim,
        ).to(device)
        metal_module.load_state_dict(metal_state)
        metal_module.eval()

        print(f"MeTAL module loaded from: {metal_state_path}")

    # Load LSLR module if it exists (frozen for evaluation)
    lslr_module: Optional[LSLRSchedule] = None

    lslr_state_path = checkpoint_dir / "lslr_state.pt"
    if lslr_state_path.exists():
        lslr_module = LSLRSchedule(
            named_params=list(theta_star.named_parameters()),
            n_steps=cfg.training.inner_steps,
            init_lr=cfg.training.inner_lr,
        ).to(device)
        lslr_module.load_state_dict(
            torch.load(lslr_state_path, map_location=device, weights_only=True)
        )
        lslr_module.eval()
        print(f"LSLR module loaded from: {lslr_state_path}")

    # iMAML: set up proximal term and inner optimizer for evaluation
    proximal_lam = 0.0
    proximal_theta: Optional[torch.Tensor] = None
    eval_inner_optimizer = "sgd"

    if cfg.training.imaml.enabled:
        im = cfg.training.imaml
        proximal_lam = im.lam

        # Load meta-learned lambda if it was saved
        lam_checkpoint = torch.load(theta_star_path, map_location=device, weights_only=False)
        saved_lam = lam_checkpoint.get("lam")
        if saved_lam is not None:
            proximal_lam = saved_lam.item()
            print(f"  Using meta-learned λ={proximal_lam:.4f}")
        else:
            print(f"  Using config λ={proximal_lam}")

        proximal_theta = torch.cat([
            p.detach().flatten() for p in theta_star.parameters()
        ]).clone()
        eval_inner_optimizer = im.inner_optimizer
        print(f"  iMAML eval: proximal_lam={proximal_lam}, inner_optimizer={eval_inner_optimizer}")

        # For iMAML+MeTAL: use n_steps=1 (one network, not per-step)
        if metal_module is not None:
            metal_module_imaml = MeTALModule(
                n_steps=1,
                n_base_params=sum(1 for _ in theta_star.parameters()),
                output_dim=net_config.output_dim,
                hidden_dim=cfg.training.metal.hidden_dim,
            ).to(device)
            metal_module_imaml.load_state_dict(metal_state)
            metal_module_imaml.eval()
            metal_module = metal_module_imaml
            print(f"  MeTAL (iMAML mode): 1 loss network")

    print()

    # =========================================================================
    # Load meta-test tasks
    # =========================================================================
    print("-" * 60)
    print("Loading meta-test tasks...")
    print("-" * 60)

    test_dir = Path(cfg.data.meta_test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Meta-test directory not found: {test_dir}")

    task_class = TASK_REGISTRY.get(pde_type)
    if task_class is None:
        raise ValueError(
            f"Unknown pde_type: {pde_type}. Available: {list(TASK_REGISTRY)}"
        )

    task_pattern = "*_fourier.npz"

    test_loader = MetaLearningDataLoader(
        test_dir, task_class=task_class, task_pattern=task_pattern, device=device
    )
    print()
    print(f"Meta-test tasks: {len(test_loader)}")
    print()

    # =========================================================================
    # Evaluation parameters
    # =========================================================================
    ev = cfg.evaluation
    k_values = ev.k_values
    noise_levels = ev.noise_levels
    fine_tune_lr = ev.fine_tune_lr
    max_steps = ev.max_steps
    fixed_steps: List[int] = ev.fixed_steps
    log_weights: bool = ev.log_weights
    loss_type: str = cfg.training.loss_function
    spectral_mode_size: int = cfg.training.spectral_loss.mode_size if cfg.training.spectral_loss.enabled else 0
    max_grad_norm: float = cfg.training.max_grad_norm
    zero_non_rhs: bool = cfg.eval_zero_non_rhs_features

    total_combos = len(test_loader) * len(k_values) * len(noise_levels)
    print("-" * 60)
    print("Evaluation grid:")
    print("-" * 60)
    print(f"  K values: {k_values}")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Fine-tune LR: {fine_tune_lr}")
    print(f"  Max steps: {max_steps}")
    print(f"  Fixed steps: {fixed_steps} ({len(fixed_steps)} snapshots)")
    print(f"  Loss function: {loss_type}")
    if spectral_mode_size > 0:
        print(f"  Spectral loss: mode_size={spectral_mode_size}")
    if max_grad_norm > 0:
        print(f"  Gradient clipping: max_norm={max_grad_norm}")
    if zero_non_rhs:
        print(f"  Zero non-RHS features: True")
    print(f"  Total evaluations: {total_combos}")
    print()

    # =========================================================================
    # Run evaluation
    # =========================================================================
    print("=" * 60)
    print("Running evaluation: w/ noisy targets on noise injected holdouts")
    print("=" * 60)
    print()

    holdout_size = ev.holdout_size

    results: Dict[str, Any] = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k_values": k_values,
            "noise_levels": noise_levels,
            "fine_tune_lr": fine_tune_lr,
            "max_steps": max_steps,
            "threshold": ev.deriv_threshold,
            "fixed_steps": fixed_steps,
            "holdout_size": holdout_size,
            "pde_type": pde_type,
        },
        "tasks": {},
    }

    # Track per-combo stats by IC type for summary
    combo_stats_by_ic: Dict[str, list] = {}

    for task_idx, task in enumerate(test_loader.tasks):
        print(f"[{task_idx + 1}/{len(test_loader)}] Task: {task.task_name}")

        task_result = evaluate_task(
            task=task,
            theta_star=theta_star,
            theta_0=theta_0,
            k_values=k_values,
            noise_levels=noise_levels,
            fine_tune_lr=fine_tune_lr,
            max_steps=max_steps,
            device=device,
            seed=seed + task_idx * 10000,
            holdout_size=holdout_size,
            fixed_steps=fixed_steps,
            metal=metal_module,
            lslr=lslr_module,
            loss_type=loss_type,
            spectral_mode_size=spectral_mode_size,
            max_grad_norm=max_grad_norm,
            log_weights=log_weights,
            zero_non_rhs_features=zero_non_rhs,
            proximal_lam=proximal_lam,
            proximal_theta=proximal_theta,
            inner_optimizer=eval_inner_optimizer,
        )

        # Serialize to NPZ
        task_npz = task_result.to_npz_dict()

        # NaN guard: check all arrays from this task before saving
        nan_keys = [
            k for k, v in task_npz.items()
            if isinstance(v, np.ndarray) and v.dtype.kind == "f" and np.isnan(v).any()
        ]
        if nan_keys:
            print(f"\nFATAL: NaN detected in {len(nan_keys)} arrays for {task.task_name}:")
            for k in nan_keys[:5]:
                print(f"  - {k}")
            if len(nan_keys) > 5:
                print(f"  ... and {len(nan_keys) - 5} more")
            print(f"Cleaning up evaluation directory...")
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
                print(f"Deleted: {eval_dir}")
            sys.exit(1)

        # Save task metadata to results dict
        results["tasks"][task.task_name] = task_result.to_json_dict()

        # Save samples to NPZ
        if task_npz:
            samples_path = samples_dir / f"{task.task_name}.npz"
            np.savez_compressed(samples_path, **task_npz)

        # Collect per-combo stats by IC type (loss + coefficient recovery)
        ic_type = task_result.ic_type
        if ic_type not in combo_stats_by_ic:
            combo_stats_by_ic[ic_type] = []

        for combo in task_result.combos:
            loss_worse_any = len(combo.worse.loss_steps) > 0
            coeff_worse_any = any(
                len(s) > 0 for s in combo.worse.coeff_steps.values()
            )

            entry: Dict[str, Any] = {
                "task": task.task_name,
                "k": combo.k,
                "noise": combo.noise,
                "maml_loss": float(combo.maml.fine_tune.holdout_losses[-1]),
                "baseline_loss": float(combo.baseline.fine_tune.holdout_losses[-1]),
                "loss_worse": loss_worse_any,
                "coeff_worse": coeff_worse_any,
            }

            maml_err = combo.maml.coefficient_recovery.error_pct
            baseline_err = combo.baseline.coefficient_recovery.error_pct
            if maml_err:
                entry["maml_coeff_error"] = maml_err[-1]
            if baseline_err:
                entry["baseline_coeff_error"] = baseline_err[-1]

            combo_stats_by_ic[ic_type].append(entry)

        print()

    # Save metadata JSON
    results_path = eval_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metadata to: {results_path}")
    print(f"Saved samples to: {samples_dir}")
    print()

    # Print evaluation summary by IC type (loss + coefficient recovery)
    print("-" * 60)
    print("Evaluation Summary by IC Type")
    print("-" * 60)
    for ic_type, entries in sorted(combo_stats_by_ic.items()):
        n = len(entries)
        loss_worse_n = sum(1 for e in entries if e["loss_worse"])
        coeff_worse_n = sum(1 for e in entries if e["coeff_worse"])

        maml_loss_avg = np.mean([e["maml_loss"] for e in entries])
        bl_loss_avg = np.mean([e["baseline_loss"] for e in entries])

        maml_cerrs = [e["maml_coeff_error"] for e in entries if "maml_coeff_error" in e]
        bl_cerrs = [
            e["baseline_coeff_error"] for e in entries if "baseline_coeff_error" in e
        ]

        print(f"\n  {ic_type} ({n} combos):")
        print(
            f"    Loss:  MAML={maml_loss_avg:.2e}  BL={bl_loss_avg:.2e}  MAML worse: {loss_worse_n}/{n}"
        )
        if maml_cerrs:
            print(
                f"    Coeff: MAML={np.mean(maml_cerrs):.1f}%  BL={np.mean(bl_cerrs):.1f}%  MAML worse: {coeff_worse_n}/{n}"
            )

        underperformers = [e for e in entries if e["loss_worse"] or e["coeff_worse"]]
        if underperformers:
            print(f"    Underperforming ({len(underperformers)}/{n}):")
            for e in underperformers[:5]:
                flags = []
                if e["loss_worse"]:
                    flags.append("loss")
                if e["coeff_worse"]:
                    flags.append("coeff")
                print(
                    f"      {e['task']} K={e['k']} noise={e['noise']:.0%} [{','.join(flags)}]"
                )
            if len(underperformers) > 5:
                print(f"      ... and {len(underperformers) - 5} more")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print()
    print(f"Results saved to: {eval_dir}/")
    print("  - results.json (metadata)")
    print("  - samples/*.npz (per-combo arrays)")
    print()
    print("Next steps:")
    print(f"  python scripts/visualize.py --config {args.config}")
    print()

    # Save log
    sys.stdout = _tee.original  # type: ignore[assignment]
    log_path = eval_dir / "evaluate.log"
    log_path.write_text(_tee.getvalue())

    done_path.touch()

    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
