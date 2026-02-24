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
from typing import List, Dict, Any, Tuple


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

import yaml
import torch
import torch.nn.functional as F
import numpy as np

from src.networks.pde_operator_network import PDEOperatorNetwork
from src.training.task_loader import MetaLearningDataLoader, PDETask, TASK_REGISTRY
from src.evaluation.jacobian import analyze_jacobian


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
    holdout_size: int = 1000,
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
        holdout_size: Number of samples for holdout evaluation (disjoint from support)

    Returns:
        Dict with task metadata and samples dict for NPZ storage
    """
    specs = task.coefficient_specs

    task_result = {
        "task_name": task.task_name,
        "coefficients": task.diffusion_coeffs,
        "coefficient_specs": [asdict(s) for s in specs],
        "ic_type": task.ic_config.get("type", "unknown"),
        "n_samples": task.n_samples,
        "loss_maml_worse": False,
        "coeff_maml_worse": [],
    }

    # Per-combo arrays stored separately in NPZ
    samples = {}

    for k_idx, k in enumerate(k_values):
        # Evaluate Fourier ONCE per (task, K) — same collocation points for all noise levels
        k_seed = seed + k_idx * 100
        available_for_holdout = task.n_samples - k
        actual_holdout = min(holdout_size, available_for_holdout)

        support_clean, holdout_clean = task.get_support_query_split(
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

                # Per-point prediction errors on holdout set (for overlay plots)
                with torch.no_grad():
                    maml_pred = maml_model(holdout_features)
                    maml_pred_errors = (maml_pred - holdout_targets).abs().cpu().numpy()
                    baseline_pred = baseline_model(holdout_features)
                    baseline_pred_errors = (
                        (baseline_pred - holdout_targets).abs().cpu().numpy()
                    )
                samples[f"{combo_key}/maml/pred_errors"] = maml_pred_errors
                samples[f"{combo_key}/baseline/pred_errors"] = baseline_pred_errors

                # Jacobian analysis for coefficient recovery
                maml_jacobian = analyze_jacobian(
                    maml_model, holdout_features, specs, device=device
                )
                baseline_jacobian = analyze_jacobian(
                    baseline_model, holdout_features, specs, device=device
                )

                # Store Jacobian distributions
                if maml_jacobian is not None:
                    for key, arr in maml_jacobian.to_npz_dict(
                        f"{combo_key}/maml"
                    ).items():
                        samples[key] = arr
                if baseline_jacobian is not None:
                    for key, arr in baseline_jacobian.to_npz_dict(
                        f"{combo_key}/baseline"
                    ).items():
                        samples[key] = arr

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

                # Flag if MAML worse on holdout or coefficient recovery
                loss_worse = maml_holdout_final > baseline_holdout_final
                worse_coeffs = []
                if maml_jacobian is not None and baseline_jacobian is not None:
                    worse_coeffs = [
                        n
                        for n in maml_jacobian.true_values.keys()
                        if maml_jacobian.coeff_error_pct(n)
                        > baseline_jacobian.coeff_error_pct(n)
                    ]

                # Store per-combo flags
                task_result[f"worse_{combo_key}"] = {
                    "loss": loss_worse,
                    "coeff": worse_coeffs,
                }

                # Accumulate task-level flags (any combo triggers — used for directory naming)
                task_result["loss_maml_worse"] |= loss_worse
                for c in worse_coeffs:
                    if c not in task_result["coeff_maml_worse"]:
                        task_result["coeff_maml_worse"].append(c)

                flags = []
                if loss_worse:
                    flags.append("loss")
                if worse_coeffs:
                    flags.append(f"coeff:{','.join(worse_coeffs)}")

                flag_str = ""
                if len(flags) > 0:
                    flag_str = f" !!! [{','.join(flags)}]"

                print(
                    f"MAML(train={maml_train_final:.2e}, holdout={maml_holdout_final:.2e}) "
                    f"BL(train={baseline_train_final:.2e}, holdout={baseline_holdout_final:.2e}){flag_str}"
                )

            except FileNotFoundError as e:
                # Noisy file doesn't exist for this task
                print(f"SKIPPED ({e})")
                # Store error info in metadata (no samples)
                task_result[f"error_{combo_key}"] = str(e)

            except Exception as e:
                print(f"ERROR ({e})")
                task_result[f"error_{combo_key}"] = str(e)

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
    exp_dir = (
        Path("data/models") / config.get("output", {}).get("base_dir", "") / exp_name
    )

    if not exp_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_dir}\nRun train_maml.py first."
        )

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

                entry: Dict[str, Any] = {
                    "task": task.task_name,
                    "k": k,
                    "noise": noise,
                    "maml_loss": float(maml_holdout[-1]),
                    "baseline_loss": float(baseline_holdout[-1]),
                    "loss_worse": worse.get("loss", False),
                    "coeff_worse": worse.get("coeff", False),
                }

                maml_coeff = coeff_recovery.get("maml")
                baseline_coeff = coeff_recovery.get("baseline")
                if maml_coeff is not None:
                    entry["maml_coeff_error"] = maml_coeff["error_pct"]
                if baseline_coeff is not None:
                    entry["baseline_coeff_error"] = baseline_coeff["error_pct"]

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
