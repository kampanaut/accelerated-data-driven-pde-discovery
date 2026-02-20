#!/usr/bin/env python3
"""
Evaluation script for MAML vs baseline comparison.

This script:
1. Loads meta-learned θ* and initial θ₀ from checkpoints
2. Loads meta-test tasks
3. For each (task, K, noise) combination:
   - Samples K points ONCE (fair comparison)
   - Fine-tunes from θ* (MAML) and θ₀ (baseline)
   - Records full loss curves
4. Saves results.json with all curves

Usage:
    python scripts/evaluate.py --config configs/experiment.yaml
"""

import sys
import copy
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import yaml
import torch
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.networks.pde_operator_network import PDEOperatorNetwork
from src.training.task_loader import (
    MetaLearningDataLoader,
    PDETask,
    BrusselatorTask,
    FitzHughNagumoTask,
    LambdaOmegaTask,
    NavierStokesTask,
    HeatEquationTask,
    NLHeatEquationTask,
)
from src.evaluation.jacobian import analyze_jacobian_ns, analyze_jacobian_br, analyze_jacobian_heat


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
) -> Dict[str, List[float]]:
    """
    Fine-tune model and return train/holdout loss at each step.

    Args:
        model: Model to fine-tune (will be modified in-place)
        features: Training input features tensor (N, 10) on device
        targets: Training target outputs tensor (N, 2) on device
        lr: Learning rate
        max_steps: Number of gradient steps
        device: Device to run on
        holdout_features: Holdout input features tensor for generalization eval
        holdout_targets: Holdout targets tensor for generalization eval

    Returns:
        Dict with 'train_losses' and optionally 'holdout_losses'
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    x = features
    y = targets

    has_holdout = holdout_features is not None and holdout_targets is not None
    if not has_holdout:
        raise Exception("Holdout is missing! Handle this quick.")

    x_holdout = holdout_features
    y_holdout = holdout_targets

    train_losses = []
    holdout_losses = []

    for _ in range(max_steps):
        # Training loss (with gradients)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        train_losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Holdout loss (no gradients)
        if has_holdout:
            with torch.no_grad():
                pred_holdout = model(x_holdout)
                holdout_loss = F.mse_loss(pred_holdout, y_holdout)
                holdout_losses.append(holdout_loss.item())

    result = {"train_losses": train_losses}
    if has_holdout:
        result["holdout_losses"] = holdout_losses

    return result


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
    clean_targets: bool = False,
    holdout_size: int = 1000,
    pde_type: str = "ns",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate one task across all (K, noise) combinations.

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
        clean_targets: If True, use clean targets with noisy features
        holdout_size: Number of samples for holdout evaluation (disjoint from support)

    Returns:
        Dict with task metadata and curves dict for NPZ storage
    """
    task_result = {
        "task_name": task.task_name,
        "coefficients": task.diffusion_coeffs,  # {'nu': X} for NS, {'D_u': X, 'D_v': Y} for BR
        "ic_type": task.ic_config.get("type", "unknown"),
        "n_samples": task.n_samples,
    }

    # Curves stored separately for NPZ
    curves = {}

    for k_idx, k in enumerate(k_values):
        for noise_idx, noise in enumerate(noise_levels):
            combo_key = f"k_{k}_noise_{noise:.2f}"
            combo_seed = seed + k_idx * 100 + noise_idx

            print(f"    K={k:4d}, noise={noise:.0%}...", end=" ", flush=True)

            try:
                # Determine holdout size (cap at available samples minus K)
                available_for_holdout = task.n_samples - k
                actual_holdout = min(holdout_size, available_for_holdout)

                # Sample K support points + holdout set (disjoint)
                support, holdout = task.get_support_query_split(
                    K_shot=k,
                    query_size=actual_holdout,
                    seed=combo_seed,
                    noise_level=noise,
                    clean_targets=clean_targets,
                )
                features, targets = support
                holdout_features, holdout_targets = holdout

                # Fine-tune from θ* (MAML)
                maml_model = copy.deepcopy(theta_star)
                maml_result = fine_tune(
                    maml_model,
                    features,
                    targets,
                    fine_tune_lr,
                    max_steps,
                    holdout_features=holdout_features,
                    holdout_targets=holdout_targets,
                )

                # Fine-tune from θ₀ (baseline)
                baseline_model = copy.deepcopy(theta_0)
                baseline_result = fine_tune(
                    baseline_model,
                    features,
                    targets,
                    fine_tune_lr,
                    max_steps,
                    holdout_features=holdout_features,
                    holdout_targets=holdout_targets,
                )

                # Store curves for NPZ
                curves[f"{combo_key}/maml_train_losses"] = np.array(
                    maml_result["train_losses"]
                )
                curves[f"{combo_key}/maml_holdout_losses"] = np.array(
                    maml_result["holdout_losses"]
                )
                curves[f"{combo_key}/baseline_train_losses"] = np.array(
                    baseline_result["train_losses"]
                )
                curves[f"{combo_key}/baseline_holdout_losses"] = np.array(
                    baseline_result["holdout_losses"]
                )

                # Per-point prediction errors on holdout set (for overlay plots)
                with torch.no_grad():
                    maml_pred = maml_model(holdout_features)
                    maml_pred_errors = (maml_pred - holdout_targets).abs().cpu().numpy()
                    baseline_pred = baseline_model(holdout_features)
                    baseline_pred_errors = (
                        (baseline_pred - holdout_targets).abs().cpu().numpy()
                    )
                curves[f"{combo_key}/maml/pred_errors"] = maml_pred_errors
                curves[f"{combo_key}/baseline/pred_errors"] = baseline_pred_errors

                # Jacobian analysis for coefficient recovery (uses task.diffusion_coeffs property)
                coeffs = task.diffusion_coeffs
                if pde_type == "ns":
                    maml_jacobian = analyze_jacobian_ns(
                        maml_model, holdout_features, coeffs["nu"], device=device
                    )
                    baseline_jacobian = analyze_jacobian_ns(
                        baseline_model, holdout_features, coeffs["nu"], device=device
                    )
                elif pde_type in ("br", "fhn", "lo"):
                    maml_jacobian = analyze_jacobian_br(
                        maml_model,
                        holdout_features,
                        D_u_true=coeffs["D_u"],
                        D_v_true=coeffs["D_v"],
                        device=device,
                    )
                    baseline_jacobian = analyze_jacobian_br(
                        baseline_model,
                        holdout_features,
                        D_u_true=coeffs["D_u"],
                        D_v_true=coeffs["D_v"],
                        device=device,
                    )
                elif pde_type in ("heat", "nl_heat"):
                    maml_jacobian = analyze_jacobian_heat(
                        maml_model, holdout_features, coeffs["D"] if "D" in coeffs else coeffs["K"], device=device
                    )
                    baseline_jacobian = analyze_jacobian_heat(
                        baseline_model, holdout_features, coeffs["D"] if "D" in coeffs else coeffs["K"], device=device
                    )
                else:
                    raise ValueError(f"Unknown pde_type: {pde_type}.")

                # Store Jacobian distributions in curves
                if maml_jacobian is not None:
                    for key, val in maml_jacobian.to_npz_dict(f"{combo_key}/maml").items():
                        curves[key] = val
                if baseline_jacobian is not None:
                    for key, val in baseline_jacobian.to_npz_dict(
                        f"{combo_key}/baseline"
                    ).items():
                        curves[key] = val

                # Store coefficient recovery summary in task_result
                coeff_key = f"coefficient_recovery_{combo_key}"
                if maml_jacobian is not None and baseline_jacobian is not None:
                    task_result[coeff_key] = {
                        "maml": maml_jacobian.to_dict(),
                        "baseline": baseline_jacobian.to_dict(),
                    }
                else:
                    task_result[coeff_key] = {"maml": None, "baseline": None}

                maml_train_final = maml_result["train_losses"][-1]
                maml_holdout_final = maml_result["holdout_losses"][-1]
                baseline_train_final = baseline_result["train_losses"][-1]
                baseline_holdout_final = baseline_result["holdout_losses"][-1]

                # Flag if MAML worse on holdout
                flag = " !!!" if maml_holdout_final > baseline_holdout_final else ""
                print(
                    f"MAML(train={maml_train_final:.2e}, holdout={maml_holdout_final:.2e}) "
                    f"BL(train={baseline_train_final:.2e}, holdout={baseline_holdout_final:.2e}){flag}"
                )

                # Track if ANY combo has MAML worse for this task (for directory naming)
                if "any_maml_worse" not in task_result:
                    task_result["any_maml_worse"] = False
                if maml_holdout_final > baseline_holdout_final:
                    task_result["any_maml_worse"] = True

            except FileNotFoundError as e:
                # Noisy file doesn't exist for this task
                print(f"SKIPPED ({e})")
                # Store error info in metadata (no curves)
                task_result[f"error_{combo_key}"] = str(e)

            except Exception as e:
                print(f"ERROR ({e})")
                task_result[f"error_{combo_key}"] = str(e)

    return task_result, curves


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
    exp_dir = (
        Path("data/models") / config.get("output", {}).get("base_dir", "") / exp_name
    )

    if not exp_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_dir}\nRun train_maml.py first."
        )

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

    # θ* (meta-learned)
    if args.checkpoint is not None:
        theta_star_path = args.checkpoint
    else:
        theta_star_path = exp_dir / "checkpoints" / "best_model.pt"

    if not theta_star_path.exists():
        raise FileNotFoundError(f"θ* checkpoint not found: {theta_star_path}")

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
    }

    theta_star = load_model_from_checkpoint(theta_star_path, device, model_config)
    theta_0 = load_model_from_checkpoint(theta_0_path, device, model_config)

    print(f"θ* loaded from: {theta_star_path}")
    print(f"θ₀ loaded from: {theta_0_path}")
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

    if pde_type == "br":
        task_class = BrusselatorTask
    elif pde_type == "fhn":
        task_class = FitzHughNagumoTask
    elif pde_type == "lo":
        task_class = LambdaOmegaTask
    elif pde_type == "ns":
        task_class = NavierStokesTask
    elif pde_type == "heat":
        task_class = HeatEquationTask
    elif pde_type == "nl_heat":
        task_class = NLHeatEquationTask
    else:
        raise ValueError(f"Unknown pde_type: {pde_type}. Use 'br', 'fhn', 'lo', 'ns', 'heat', or 'nl_heat'.")

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

    total_combos = len(test_loader) * len(k_values) * len(noise_levels)
    print("-" * 60)
    print("Evaluation grid:")
    print("-" * 60)
    print(f"  K values: {k_values}")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Fine-tune LR: {fine_tune_lr}")
    print(f"  Max steps: {max_steps}")
    print(f"  Total evaluations: {total_combos}")
    print()

    # =========================================================================
    # Run evaluation (both noisy and clean target modes)
    # =========================================================================
    target_modes = [
        ("noisy_targets", False),  # Noisy features + noisy targets
        # ("clean_targets", True),    # Noisy features + clean targets
    ]

    for mode_name, clean_targets in target_modes:
        print("=" * 60)
        print(f"Running evaluation: {mode_name}")
        print("=" * 60)
        print()

        holdout_size = eval_cfg.get("holdout_size", 1000)

        results = {
            "experiment_name": exp_name,
            "timestamp": datetime.now().isoformat(),
            "target_mode": mode_name,
            "config": {
                "k_values": k_values,
                "noise_levels": noise_levels,
                "fine_tune_lr": fine_tune_lr,
                "max_steps": max_steps,
                "threshold": eval_cfg.get("threshold", 1e-6),
                "fixed_steps": eval_cfg.get("fixed_steps", [50, 100, 200]),
                "clean_targets": clean_targets,
                "holdout_size": holdout_size,
                "pde_type": pde_type,
            },
            "tasks": {},
        }

        # Create curves directory
        eval_dir = exp_dir / "evaluation" / mode_name
        curves_dir = eval_dir / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)

        # Track holdout losses by IC type for summary
        holdout_by_ic = {}  # ic_type -> list of (maml_final, baseline_final, task_name, combo_key)

        for task_idx, task in enumerate(test_loader.tasks):
            print(f"[{task_idx + 1}/{len(test_loader)}] Task: {task.task_name}")

            task_result, task_curves = evaluate_task(
                task=task,
                theta_star=theta_star,
                theta_0=theta_0,
                k_values=k_values,
                noise_levels=noise_levels,
                fine_tune_lr=fine_tune_lr,
                max_steps=max_steps,
                device=device,
                seed=seed + task_idx * 10000,
                clean_targets=clean_targets,
                holdout_size=holdout_size,
                pde_type=pde_type,
            )

            # Save task metadata to results dict
            results["tasks"][task.task_name] = task_result

            # Save curves to NPZ
            if task_curves:
                curves_path = curves_dir / f"{task.task_name}.npz"
                np.savez_compressed(curves_path, **task_curves)

            # Collect holdout losses by IC type
            ic_type = task_result.get("ic_type", "unknown")
            if ic_type not in holdout_by_ic:
                holdout_by_ic[ic_type] = []

            for key in task_curves.keys():
                if not key.endswith("/maml_holdout_losses"):
                    continue
                combo = key.rsplit("/", 1)[0]  # e.g., "k_100_noise_0.01"
                # Parse K and noise from combo key
                parts = combo.split("_")  # ['k', '100', 'noise', '0.01']
                k_val = int(parts[1])
                noise_val = float(parts[3])

                maml_holdout = task_curves.get(f"{combo}/maml_holdout_losses")
                baseline_holdout = task_curves.get(f"{combo}/baseline_holdout_losses")
                if maml_holdout is not None and baseline_holdout is not None:
                    holdout_by_ic[ic_type].append(
                        {
                            "maml_final": float(maml_holdout[-1]),
                            "baseline_final": float(baseline_holdout[-1]),
                            "task": task.task_name,
                            "k": k_val,
                            "noise": noise_val,
                        }
                    )

            print()

        # Save metadata JSON
        results_path = eval_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved metadata to: {results_path}")
        print(f"Saved curves to: {curves_dir}")
        print()

        # Print holdout summary by IC type
        print("-" * 60)
        print("Holdout Loss Summary by IC Type")
        print("-" * 60)
        for ic_type, entries in sorted(holdout_by_ic.items()):
            maml_losses = [e["maml_final"] for e in entries]
            baseline_losses = [e["baseline_final"] for e in entries]
            maml_mean = np.mean(maml_losses)
            baseline_mean = np.mean(baseline_losses)

            # Flag if MAML worse than baseline on average
            flag = " *** MAML WORSE ***" if maml_mean > baseline_mean else ""
            print(
                f"  {ic_type}: MAML={maml_mean:.2e}, Baseline={baseline_mean:.2e}{flag}"
            )

            # List individual cases where MAML underperforms
            underperformers = [
                e for e in entries if e["maml_final"] > e["baseline_final"]
            ]
            if underperformers:
                print(
                    f"    Underperforming cases ({len(underperformers)}/{len(entries)}):"
                )
                for e in underperformers[:5]:  # Show at most 5
                    print(
                        f"      {e['task']} K={e['k']} noise={e['noise']:.0%}: MAML={e['maml_final']:.2e} > Baseline={e['baseline_final']:.2e}"
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
    print("Results saved to:")
    for mode_name, _ in target_modes:
        print(f"  {exp_dir / 'evaluation' / mode_name}/")
        print("    - results.json (metadata)")
        print("    - curves/*.npz (loss curves)")
    print()
    print("Next steps:")
    print(f"  python scripts/visualize.py --config {args.config}")
    print()


if __name__ == "__main__":
    main()
