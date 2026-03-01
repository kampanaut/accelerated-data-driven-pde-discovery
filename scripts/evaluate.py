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
import argparse
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Callable, Optional


from numpy.typing import NDArray
import yaml
import torch
import torch.nn.functional as F
import numpy as np

from src.networks.pde_operator_network import PDEOperatorNetwork
from src.training.task_loader import MetaLearningDataLoader, PDETask, TASK_REGISTRY
from src.training.maml import MeTALModule
from src.training.spectral_loss import compute_spectral_loss
from src.evaluation.jacobian import analyze_jacobian, JacobianResults
from src.evaluation.metrics import compress_step_ranges


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


def load_config(config_path: Path) -> dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(
    checkpoint_path: Path, device: str, model_config: dict
) -> PDEOperatorNetwork:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model to
        model_config: Optional model architecture config. If None, tries to extract
                     from checkpoint (works for initial_model.pt format)

    Returns:
        Loaded PDEOperatorNetwork
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = PDEOperatorNetwork(
        input_dim=model_config.get("input_dim", 10),
        output_dim=model_config.get("output_dim", 2),
        hidden_dims=model_config.get("hidden_dims", [100, 100]),
        activation=model_config.get("activation", "tanh"),
        conv_filters=model_config.get("conv_filters", 0),
        conv_kernel_size=model_config.get("conv_kernel_size", 3),
    )

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
    loss_type: str = "normalized_mse",
    coords: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    Lx: float = 0.0,
    Ly: float = 0.0,
    spectral_mode_size: int = 0,
    max_grad_norm: float = 0.0,
) -> Dict[str, List[float]]:
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
        fixed_steps: Steps at which to call on_step (1-indexed, e.g. [1, 10, 50])
        on_step: Callback called at each fixed_step with (model, step_number)
        metal: Frozen MeTALModule (None = standard loss)
        loss_type: Loss function — 'mse', 'normalized_mse', or 'mae'

    Returns:
        Dict with 'train_losses' and 'holdout_losses'
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

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

    # Wrap with spectral loss if coords provided and mode_size > 0
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

    # Step 0: snapshot θ* before any gradient update
    if on_step is not None and 0 in fixed_steps_set:
        on_step(model, 0)
        model.train()

    def _step(step_idx: int, grad_loss: torch.Tensor) -> None:
        """Shared step logic: backward, optimize, holdout, callback."""
        opt.zero_grad()
        grad_loss.backward()
        # Gradient clipping (Qin & Beatson 2022: max_norm=100.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

        with torch.no_grad():
            pred_holdout = model(x_holdout)
            holdout_loss = cost_fn(pred_holdout, y_holdout)
            holdout_losses.append(holdout_loss.item())

        step_num = step_idx + 1
        if on_step is not None and step_num in fixed_steps_set:
            on_step(model, step_num)
            model.train()

    # Phase 1: MeTAL-shaped inner steps (if MeTAL provided)
    metal_steps = metal.n_steps if metal is not None else 0
    for i in range(min(metal_steps, max_steps)):
        pred = model(x)
        recorded_loss = cost_fn(pred, y)
        train_losses.append(recorded_loss.item())

        meta_support = metal.support_step(i, model, pred, y, cost_fn)  # type: ignore[union-attr]
        meta_query = metal.query_step(i, model, pred)  # type: ignore[union-attr]
        _step(i, recorded_loss + meta_support + meta_query)

    # Phase 2: Standard loss for remaining steps
    for i in range(metal_steps, max_steps):
        pred = model(x)
        recorded_loss = cost_fn(pred, y)
        train_losses.append(recorded_loss.item())

        _step(i, recorded_loss)

    return {"train_losses": train_losses, "holdout_losses": holdout_losses}


def _snapshots_to_dict(snapshots: List[JacobianResults]) -> Dict[str, Any]:
    """Convert list of per-step JacobianResults to array-indexed dict for JSON."""
    d: Dict[str, Any] = {}
    for name in snapshots[0].estimates:
        d[f"{name}_true"] = snapshots[0].true_values[name]
        d[f"{name}_recovered"] = [jac.recovered(name) for jac in snapshots]
        d[f"{name}_error_pct"] = [jac.coeff_error_pct(name) for jac in snapshots]
        d[f"{name}_mean"] = [float(np.mean(jac.estimates[name])) for jac in snapshots]
        d[f"{name}_std"] = [float(np.std(jac.estimates[name])) for jac in snapshots]
    d["error_pct"] = [
        float(np.mean([jac.coeff_error_pct(n) for n in jac.true_values]))
        for jac in snapshots
    ]
    return d


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
    loss_type: str = "normalized_mse",
    spectral_mode_size: int = 0,
    max_grad_norm: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate one task across all (K, noise) combinations.

    Extracts Jacobian and pred_errors at every fixed_step during fine-tuning.

    Args:
        task: The task to evaluate
        theta_star: Meta-learned model (θ*)
        theta_0: Initial model (θ₀)
        k_values: List of K values to test
        noise_levels: List of noise levels to test
        fine_tune_lr: Learning rate for fine-tuning
        max_steps: Number of fine-tuning steps
        device: Device to run on
        seed: Base random seed
        holdout_size: Number of samples for holdout evaluation (disjoint from support)
        fixed_steps: Steps at which to extract Jacobian + pred_errors

    Returns:
        Dict with task metadata and samples dict for NPZ storage
    """
    specs = task.coefficient_specs

    task_result: Dict[str, Any] = {
        "task_name": task.task_name,
        "coefficients": task.diffusion_coeffs,
        "coefficient_specs": [asdict(s) for s in specs],
        "ic_type": task.ic_config.get("type", "unknown"),
        "n_samples": task.n_samples,
        "loss_worse_steps": [],
        "coeff_worse_steps": {},
    }

    # Per-combo arrays stored separately in NPZ
    samples: Dict[str, Any] = {}

    # Track combo metadata for best-combo selection after the loop
    combo_tracker: List[Dict[str, Any]] = []

    for k_idx, k in enumerate(k_values):
        # Evaluate Fourier ONCE per (task, K) — same collocation points for all noise levels
        k_seed = seed + k_idx * 100
        available_for_holdout = task.n_samples - k
        actual_holdout = min(holdout_size, available_for_holdout)

        support_clean, holdout_clean, support_coords, holdout_coords = task.get_support_query_split(
            K_shot=k,
            query_size=actual_holdout,
            seed=k_seed,
        )

        for noise_idx, noise in enumerate(noise_levels):
            combo_key = f"k_{k}_noise_{noise:.2f}"

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

                # --- Build callback for intermediate Jacobian extraction ---
                maml_jac_snapshots: List[JacobianResults] = []
                maml_pred_snapshots: List[NDArray[np.integer[Any]]] = []
                baseline_jac_snapshots: List[JacobianResults] = []
                baseline_pred_snapshots: List[NDArray[np.integer[Any]]] = []

                def _make_on_step(
                    jac_list: List[JacobianResults],
                    pred_list: List[np.ndarray],
                    h_feat: torch.Tensor,
                    h_tgt: torch.Tensor,
                ) -> Callable[[torch.nn.Module, int], None]:
                    def on_step(model: torch.nn.Module, _: int) -> None:
                        jac = analyze_jacobian(model, h_feat, specs, device=device)
                        jac_list.append(jac)
                        with torch.no_grad():
                            pred = model(h_feat)
                            pred_errors = (pred - h_tgt).abs().cpu().numpy()
                        pred_list.append(pred_errors)

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
                    ),
                    metal=metal,
                    loss_type=loss_type,
                    coords=support_coords,
                    Lx=task.Lx,
                    Ly=task.Ly,
                    spectral_mode_size=spectral_mode_size,
                    max_grad_norm=max_grad_norm,
                )

                # Fine-tune from θ₀ (baseline) — same loss, no MeTAL
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
                    ),
                    loss_type=loss_type,
                    coords=support_coords,
                    Lx=task.Lx,
                    Ly=task.Ly,
                    spectral_mode_size=spectral_mode_size,
                    max_grad_norm=max_grad_norm,
                )

                # Store curves for NPZ
                samples[f"{combo_key}/maml_train_losses"] = np.array(
                    maml_result["train_losses"]
                )
                samples[f"{combo_key}/maml_holdout_losses"] = np.array(
                    maml_result["holdout_losses"]
                )
                samples[f"{combo_key}/baseline_train_losses"] = np.array(
                    baseline_result["train_losses"]
                )
                samples[f"{combo_key}/baseline_holdout_losses"] = np.array(
                    baseline_result["holdout_losses"]
                )

                # Stack per-step Jacobian distributions into NPZ
                # Shape: (n_fixed_steps, holdout_size)
                for label, jac_snaps, pred_snaps in [
                    ("maml", maml_jac_snapshots, maml_pred_snapshots),
                    ("baseline", baseline_jac_snapshots, baseline_pred_snapshots),
                ]:
                    for coeff_name in jac_snaps[0].estimates.keys():
                        stacked = np.stack(
                            [jac.estimates[coeff_name] for jac in jac_snaps]
                        )
                        samples[f"{combo_key}/{label}/{coeff_name}"] = stacked
                        samples[f"{combo_key}/{label}/{coeff_name}_true"] = np.array(
                            [jac_snaps[0].true_values[coeff_name]]  # We just get 1
                            # arbitrary snapshot, since all true_values are the same
                        )
                    # pred_errors: (n_fixed_steps, holdout_size, 2)
                    samples[f"{combo_key}/{label}/pred_errors"] = np.stack(pred_snaps)

                # Store coefficient recovery summary in task_result (arrays)
                coeff_key = f"coefficient_recovery_{combo_key}"
                task_result[coeff_key] = {
                    "fixed_steps": fixed_steps,
                    "maml": _snapshots_to_dict(maml_jac_snapshots),
                    "baseline": _snapshots_to_dict(baseline_jac_snapshots),
                }

                maml_train_final = maml_result["train_losses"][-1]
                maml_holdout_final = maml_result["holdout_losses"][-1]
                baseline_train_final = baseline_result["train_losses"][-1]
                baseline_holdout_final = baseline_result["holdout_losses"][-1]

                # Per-step worse comparison
                loss_worse_steps: List[int] = []
                coeff_worse_steps: Dict[str, List[int]] = {
                    n: [] for n in maml_jac_snapshots[0].true_values.keys()
                }

                for si, step in enumerate(fixed_steps):
                    # Loss: compare holdout at this step
                    maml_h = maml_result["holdout_losses"][step - 1]
                    baseline_h = baseline_result["holdout_losses"][step - 1]
                    if maml_h > baseline_h:
                        loss_worse_steps.append(step)

                    # Coefficients: compare error at this step
                    for n in maml_jac_snapshots[si].true_values:
                        if maml_jac_snapshots[si].coeff_error_pct(
                            n
                        ) > baseline_jac_snapshots[si].coeff_error_pct(n):
                            coeff_worse_steps[n].append(step)

                # Store per-combo flags (step-granular)
                task_result[f"worse_{combo_key}"] = {
                    "loss_steps": loss_worse_steps,
                    "coeff_steps": coeff_worse_steps,
                }

                # Accumulate task-level flags (union across combos)
                for step in loss_worse_steps:
                    if step not in task_result["loss_worse_steps"]:
                        task_result["loss_worse_steps"].append(step)
                for n, steps in coeff_worse_steps.items():
                    if n not in task_result["coeff_worse_steps"]:
                        task_result["coeff_worse_steps"][n] = []
                    for step in steps:
                        if step not in task_result["coeff_worse_steps"][n]:
                            task_result["coeff_worse_steps"][n].append(step)

                # Print summary
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

                # Track for best-combo selection
                coeff_recovery = task_result[f"coefficient_recovery_{combo_key}"]
                combo_tracker.append({
                    "combo_key": combo_key,
                    "k": k,
                    "k_idx": k_idx,
                    "noise": noise,
                    "noise_idx": noise_idx,
                    "error_pct_array": coeff_recovery["maml"]["error_pct"],
                })

            except FileNotFoundError as e:
                # Noisy file doesn't exist for this task
                print(f"SKIPPED ({e})")
                # Store error info in metadata (no samples)
                task_result[f"error_{combo_key}"] = str(e)

            except Exception as e:
                print(f"ERROR ({e})")
                task_result[f"error_{combo_key}"] = str(e)

    # ─── Best-combo prediction capture ─────────────────────────────────────
    # Re-fine-tune the best combo and record model predictions at each step
    # for spatial visualization of how the model adapts.
    if combo_tracker:
        # Pick combo whose best snapshot has lowest mean coefficient error
        best = min(combo_tracker, key=lambda c: min(c["error_pct_array"]))
        best_k = best["k"]
        best_k_idx = best["k_idx"]
        best_noise = best["noise"]
        best_noise_idx = best["noise_idx"]
        best_combo_key = best["combo_key"]
        best_k_seed = seed + best_k_idx * 100

        print(f"    Best combo: {best_combo_key} (min error={min(best['error_pct_array']):.1f}%)")

        # Re-create the same split (deterministic seed)
        available_for_holdout = task.n_samples - best_k
        actual_holdout = min(holdout_size, available_for_holdout)
        s_clean, h_clean, s_coords, h_coords = task.get_support_query_split(
            K_shot=best_k,
            query_size=actual_holdout,
            seed=best_k_seed,
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

        # Step-0 prediction (θ* before any fine-tuning)
        bc_model = copy.deepcopy(theta_star)
        pred_snapshots: List[NDArray[np.floating[Any]]] = []
        with torch.no_grad():
            pred_snapshots.append(bc_model(bc_h_features).cpu().numpy())

        # Prediction callback for fixed_steps
        def _pred_on_step(model: torch.nn.Module, _: int) -> None:
            with torch.no_grad():
                pred_snapshots.append(model(bc_h_features).cpu().numpy())

        # Re-fine-tune (MAML model only — baseline not needed for this plot)
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
            loss_type=loss_type,
            coords=s_coords,
            Lx=task.Lx,
            Ly=task.Ly,
            spectral_mode_size=spectral_mode_size,
            max_grad_norm=max_grad_norm,
        )

        # Retrieve per-step coeff errors from already-computed results
        coeff_recovery = task_result[f"coefficient_recovery_{best_combo_key}"]
        step_errors = coeff_recovery["maml"]["error_pct"]

        # Save to NPZ
        all_steps = [0] + fixed_steps
        samples["best_combo/key"] = np.array(best_combo_key)
        samples["best_combo/predictions"] = np.stack(pred_snapshots)
        samples["best_combo/true_targets"] = bc_h_targets.cpu().numpy()
        samples["best_combo/x_pts"] = h_coords[0].cpu().numpy()
        samples["best_combo/y_pts"] = h_coords[1].cpu().numpy()
        samples["best_combo/steps"] = np.array(all_steps)
        samples["best_combo/coeff_error"] = np.array(
            [float("nan")] + step_errors  # step 0 has no coeff estimate
        )

    return task_result, samples


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
        help="Path to θ* checkpoint (default: best_model.pt in experiment dir)",
    )
    args = parser.parse_args()

    # =========================================================================
    # Load configuration
    # =========================================================================
    print("=" * 60)
    print("MAML Evaluation: θ* vs θ₀ Comparison")
    print("=" * 60)
    print()

    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    base_dir = Path(config.get("output", {}).get("base_dir", "data/models"))
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

    # Tee stdout to capture all print output for log file
    _tee = _TeeStream(sys.stdout)
    sys.stdout = _tee  # type: ignore[assignment]

    print(f"Experiment: {exp_name}")
    print(f"Experiment directory: {exp_dir}")
    print()

    # =========================================================================
    # Setup
    # =========================================================================
    seed = config["experiment"].get("seed", 42)
    device = config["experiment"].get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    pde_type = config["experiment"].get("pde_type", "ns")  # 'ns' or 'br'

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
        best_path = exp_dir / "checkpoints" / "best_model.pt"
        latest_path = exp_dir / "checkpoints" / "latest_model.pt"
        if best_path.exists():
            theta_star_path = best_path
        elif latest_path.exists():
            theta_star_path = latest_path
            print(f"best_model.pt not found — using latest_model.pt")
        else:
            print(f"No checkpoint found in {exp_dir / 'checkpoints'}")
            sys.exit(1)

    # θ₀ (initial)
    theta_0_path = exp_dir / "checkpoints" / "initial_model.pt"
    if not theta_0_path.exists():
        raise FileNotFoundError(f"θ₀ checkpoint not found: {theta_0_path}")

    # Get model architecture from training config
    train_cfg = config.get("training", {})
    model_config = {
        "input_dim": train_cfg.get("input_dim", 10),
        "output_dim": train_cfg.get("output_dim", 2),
        "hidden_dims": train_cfg.get("hidden_dims", [100, 100]),
        "activation": train_cfg.get("activation", "tanh"),
        "conv_filters": train_cfg.get("conv_filters", 0),
        "conv_kernel_size": train_cfg.get("conv_kernel_size", 3),
    }

    theta_star = load_model_from_checkpoint(theta_star_path, device, model_config)
    theta_0 = load_model_from_checkpoint(theta_0_path, device, model_config)

    print(f"θ* loaded from: {theta_star_path}")
    print(f"θ₀ loaded from: {theta_0_path}")

    # Load MeTAL module if it exists (frozen for evaluation)
    metal_module: Optional[MeTALModule] = None

    checkpoint_dir = theta_star_path.parent
    metal_state_path = checkpoint_dir / "metal_state.pt"

    if metal_state_path.exists():
        n_base_params = sum(1 for _ in theta_star.parameters())
        output_dim = model_config.get("output_dim", 2)
        inner_steps = config.get("training", {}).get("inner_steps", 1)
        metal_hidden_dim = (
            config.get("training", {}).get("metal", {}).get("hidden_dim", 64)
        )

        metal_module = MeTALModule(
            n_steps=inner_steps,
            n_base_params=n_base_params,
            output_dim=output_dim,
            hidden_dim=metal_hidden_dim,
        ).to(device)
        metal_module.load_state_dict(
            torch.load(metal_state_path, map_location=device, weights_only=True)
        )
        metal_module.eval()

        print(f"MeTAL module loaded from: {metal_state_path}")

    print()

    # =========================================================================
    # Load meta-test tasks
    # =========================================================================
    print("-" * 60)
    print("Loading meta-test tasks...")
    print("-" * 60)

    test_dir = Path(config["data"]["meta_test_dir"])
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
    eval_cfg = config["evaluation"]
    k_values = eval_cfg.get("k_values", [10, 50, 100, 500, 1000])
    noise_levels = eval_cfg.get("noise_levels", [0.0, 0.01, 0.05, 0.10])
    fine_tune_lr = eval_cfg.get("fine_tune_lr", 0.01)
    max_steps = eval_cfg.get("max_steps", 1000)
    fixed_steps: List[int] = eval_cfg.get("fixed_steps", [50, 100, 200])
    loss_type: str = train_cfg.get("loss_function", "normalized_mse")
    spectral_cfg = train_cfg.get("spectral_loss", {})
    spectral_mode_size: int = spectral_cfg.get("mode_size", 0) if spectral_cfg.get("enabled", False) else 0
    max_grad_norm: float = train_cfg.get("max_grad_norm", 0.0)

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
    print(f"  Total evaluations: {total_combos}")
    print()

    # =========================================================================
    # Run evaluation
    # =========================================================================
    print("=" * 60)
    print("Running evaluation: w/ noisy targets on noise injected holdouts")
    print("=" * 60)
    print()

    holdout_size = eval_cfg.get("holdout_size", 1000)

    results: Dict[str, Any] = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k_values": k_values,
            "noise_levels": noise_levels,
            "fine_tune_lr": fine_tune_lr,
            "max_steps": max_steps,
            "threshold": eval_cfg.get("threshold", 1e-6),
            "fixed_steps": eval_cfg.get("fixed_steps", [50, 100, 200]),
            "holdout_size": holdout_size,
            "pde_type": pde_type,
        },
        "tasks": {},
    }

    # Create samples directory
    eval_dir = exp_dir / "evaluation"
    samples_dir = eval_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Track per-combo stats by IC type for summary
    combo_stats_by_ic: Dict[str, list] = {}

    for task_idx, task in enumerate(test_loader.tasks):
        print(f"[{task_idx + 1}/{len(test_loader)}] Task: {task.task_name}")

        task_result, task_samples = evaluate_task(
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
            loss_type=loss_type,
            spectral_mode_size=spectral_mode_size,
            max_grad_norm=max_grad_norm,
        )

        # Save task metadata to results dict
        results["tasks"][task.task_name] = task_result

        # Save samples to NPZ
        if task_samples:
            samples_path = samples_dir / f"{task.task_name}.npz"
            np.savez_compressed(samples_path, **task_samples)

        # Collect per-combo stats by IC type (loss + coefficient recovery)
        ic_type = task_result.get("ic_type", "unknown")
        if ic_type not in combo_stats_by_ic:
            combo_stats_by_ic[ic_type] = []

        for k in k_values:
            for noise in noise_levels:
                combo_key = f"k_{k}_noise_{noise:.2f}"

                maml_holdout = task_samples.get(f"{combo_key}/maml_holdout_losses")
                baseline_holdout = task_samples.get(
                    f"{combo_key}/baseline_holdout_losses"
                )
                if maml_holdout is None or baseline_holdout is None:
                    continue

                worse = task_result.get(f"worse_{combo_key}", {})
                coeff_recovery = task_result.get(
                    f"coefficient_recovery_{combo_key}", {}
                )

                loss_worse_any = len(worse.get("loss_steps", [])) > 0
                coeff_worse_any = any(
                    len(s) > 0 for s in worse.get("coeff_steps", {}).values()
                )
                entry: Dict[str, Any] = {
                    "task": task.task_name,
                    "k": k,
                    "noise": noise,
                    "maml_loss": float(maml_holdout[-1]),
                    "baseline_loss": float(baseline_holdout[-1]),
                    "loss_worse": loss_worse_any,
                    "coeff_worse": coeff_worse_any,
                }

                maml_coeff = coeff_recovery.get("maml")
                baseline_coeff = coeff_recovery.get("baseline")
                if maml_coeff is not None:
                    entry["maml_coeff_error"] = maml_coeff["error_pct"][-1]
                if baseline_coeff is not None:
                    entry["baseline_coeff_error"] = baseline_coeff["error_pct"][-1]

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
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
