#!/usr/bin/env python3
"""
MAML meta-training for PDE discovery.

This script:
1. Loads experiment configuration from YAML
2. Loads meta-train and meta-val task datasets
3. Saves initial model weights (θ₀) for baseline comparison
4. Runs MAML meta-training via MAMLTrainer
5. Saves best model checkpoint (θ*) and training history

Usage:
    python scripts/train_maml.py --config configs/experiment.yaml
    python scripts/train_maml.py --config configs/experiment.yaml --resume
"""

import io
import re
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


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


import torch
torch.set_num_threads(9)
torch.set_num_interop_threads(4)

from src.config import ExperimentConfig
from src.networks.pde_operator_network import PDEOperatorNetwork
from src.training.task_loader import MetaLearningDataLoader, TASK_REGISTRY
from src.training.maml import MAMLTrainer


def setup_output_dirs(cfg: ExperimentConfig) -> Path:
    """Create experiment output directory structure.

    Resolves suffixed directories: if the exact name doesn't exist,
    looks for ENDNAN-suffixed dirs (usable) or ISNAN-suffixed dirs (skip).
    """
    base_dir = Path(cfg.output.base_dir)
    exp_name = cfg.experiment.name
    exp_dir = base_dir / exp_name

    if not exp_dir.exists():
        candidates = sorted(base_dir.glob(f"{exp_name}-*"))
        endnan = [d for d in candidates if re.search(r"-ENDNAN@\d+$", d.name)]
        isnan = [d for d in candidates if d.name.endswith("-ISNAN")]

        if endnan:
            exp_dir = endnan[0]
            print(f"Using flagged directory: {exp_dir.name}")
        elif isnan:
            print(f"Skipping {exp_name}: flagged as {isnan[0].name}")
            sys.exit(0)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "training").mkdir(parents=True, exist_ok=True)

    return exp_dir


def main():
    parser = argparse.ArgumentParser(
        description="MAML meta-training for PDE discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    args = parser.parse_args()

    # =========================================================================
    # Load configuration
    # =========================================================================
    print("=" * 60)
    print("MAML Meta-Training for PDE Discovery")
    print("=" * 60)
    print()

    cfg = ExperimentConfig.from_yaml(args.config)
    exp_dir = setup_output_dirs(cfg)

    # Tee stdout to capture all print output for log file
    _tee = _TeeStream(sys.stdout)
    sys.stdout = _tee  # type: ignore[assignment]

    print(f"Experiment: {cfg.experiment.name}")
    print(f"Output directory: {exp_dir}")
    print()

    # Save config copy
    config_copy_path = exp_dir / "training" / "config.yaml"
    shutil.copy(args.config, config_copy_path)

    # Guard: DONE sentinel means training completed — never overwrite
    done_path = exp_dir / "training" / "DONE"
    if done_path.exists():
        print(f"SKIP: {exp_dir.name} already trained (DONE sentinel exists).")
        sys.exit(0)

    # =========================================================================
    # Set random seeds
    # =========================================================================
    seed = cfg.experiment.seed
    device = cfg.experiment.device
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()

    # =========================================================================
    # Load datasets
    # =========================================================================
    print("-" * 60)
    print("Loading datasets...")
    print("-" * 60)

    train_dir = Path(cfg.data.meta_train_dir)
    val_dir = Path(cfg.data.meta_val_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Meta-train directory not found: {train_dir}")

    # Select task class based on PDE type
    pde_type = cfg.experiment.pde_type
    task_class = TASK_REGISTRY.get(pde_type)
    if task_class is None:
        raise ValueError(
            f"Unknown pde_type: {pde_type}. Available: {list(TASK_REGISTRY)}"
        )
    print(f"PDE type: {pde_type} ({task_class.__name__})")

    task_pattern = "*_fourier.npz"

    train_loader = MetaLearningDataLoader(
        train_dir, task_class=task_class, task_pattern=task_pattern, device=device
    )

    # Val loader only needed for patience mode (early stopping)
    val_loader = None
    if cfg.training.patience > 0:
        if not val_dir.exists():
            raise FileNotFoundError(f"Meta-val directory not found: {val_dir}")
        val_loader = MetaLearningDataLoader(
            val_dir, task_class=task_class, task_pattern=task_pattern, device=device
        )

    print()
    print(f"Meta-train tasks: {len(train_loader)}")
    if val_loader is not None:
        print(f"Meta-val tasks: {len(val_loader)}")
    print()

    # =========================================================================
    # Create model
    # =========================================================================
    print("-" * 60)
    print("Creating model...")
    print("-" * 60)

    net_config = cfg.to_network_config()
    model = PDEOperatorNetwork(net_config)

    weight_init = cfg.training.weight_init
    if weight_init == "zeros":
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        print(f"Weight init: zeros ({sum(p.numel() for p in model.parameters())} params zeroed)")
    elif weight_init == "expected":
        # Compute E[coeff] from training tasks, set weights to match PDE structure
        # For Heat: weights = [0, 0, 0, E[D], E[D]] (features are [u, u_x, u_y, u_xx, u_yy])
        specs = train_loader.tasks[0].coefficient_specs
        coeff_means = {}
        for spec in specs:
            values = [t.coefficient_specs[i].true_value
                      for t in train_loader.tasks
                      for i, s in enumerate(t.coefficient_specs)
                      if s.name == spec.name]
            coeff_means[spec.name] = sum(values) / len(values)

        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            # Set weights at perturb_indices to E[coeff] for each spec
            for spec in specs:
                mean_val = coeff_means[spec.name]
                for p in model.parameters():
                    if p.dim() >= 2:  # weight matrix (out, in)
                        for idx in spec.perturb_indices:
                            p[spec.output_index, idx] = mean_val
                    elif p.dim() == 1 and len(spec.perturb_indices) == 0:
                        # bias case — skip for now
                        pass

        weight_vec = torch.cat([p.flatten() for p in model.parameters()])
        coeff_str = ", ".join(f"{k}={v:.4f}" for k, v in coeff_means.items())
        print(f"Weight init: expected ({coeff_str})")
        print(f"  Weights: {weight_vec.tolist()}")

    print(model)
    print()

    # =========================================================================
    # Save initial weights (θ₀) for baseline comparison
    # =========================================================================
    initial_checkpoint_path = exp_dir / "checkpoints" / "initial_model.pt"

    if not args.resume:
        # Guard against accidental overwrite of a trained or interrupted experiment
        for guard_name in ("final_model.pt", "latest_model.pt"):
            guard_path = exp_dir / "checkpoints" / guard_name
            if guard_path.exists():
                print(f"ERROR: {guard_path} already exists.")
                print(
                    "  Use --resume to continue training, or delete the experiment directory first."
                )
                sys.exit(1)

        print("-" * 60)
        print("Saving initial weights (θ₀)...")
        print("-" * 60)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": net_config.to_dict(),
                "timestamp": datetime.now().isoformat(),
            },
            initial_checkpoint_path,
        )

        print(f"Saved to: {initial_checkpoint_path}")
        print()
    else:
        print(f"Resume mode: No need to regenerate {initial_checkpoint_path}")
        print()

    # =========================================================================
    # Create MAML trainer
    # =========================================================================
    print("-" * 60)
    print("Configuring MAML trainer...")
    print("-" * 60)

    maml_config = cfg.to_maml_config()

    trainer = MAMLTrainer(
        model=model,
        config=maml_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Resume from checkpoint if requested
    latest_checkpoint = exp_dir / "checkpoints" / "latest_model.pt"
    if args.resume and latest_checkpoint.exists():
        print(f"Resuming from: {latest_checkpoint}")
        trainer.load_checkpoint(latest_checkpoint)
    print()

    # =========================================================================
    # Run training
    # =========================================================================
    print("-" * 60)
    print("Starting MAML training...")
    print("-" * 60)
    print()

    checkpoint_dir = exp_dir / "checkpoints"
    history, done = trainer.train(checkpoint_dir=checkpoint_dir)

    # =========================================================================
    # Post-training directory rename
    # =========================================================================
    nan_iter = trainer.pop_nan_iteration()
    isnan_threshold = 20
    if nan_iter is not None:
        # NaN exit — ISNAN if too early (useless), ENDNAN if partial (salvageable)
        if nan_iter < isnan_threshold:
            suffix = "-ISNAN"
        else:
            suffix = f"-ENDNAN@{nan_iter}"
        if not re.search(r"-(ENDNAN@\d+|ISNAN)$", exp_dir.name):
            new_dir = exp_dir.parent / f"{exp_dir.name}{suffix}"
            exp_dir.rename(new_dir)
            exp_dir = new_dir
            print(f"Renamed → {exp_dir.name}")
    else:
        # Normal exit — strip ENDNAN/ISNAN suffix if present
        if re.search(r"-(ENDNAN@\d+|ISNAN)$", exp_dir.name):
            clean_name = re.sub(r"-(ENDNAN@\d+|ISNAN)$", "", exp_dir.name)
            new_dir = exp_dir.parent / clean_name
            exp_dir.rename(new_dir)
            exp_dir = new_dir
            print(f"Renamed → {exp_dir.name}")

    checkpoint_dir = exp_dir / "checkpoints"

    # =========================================================================
    # Save training history
    # =========================================================================
    print()
    print("-" * 60)
    print("Saving training history...")
    print("-" * 60)

    history_path = exp_dir / "training" / "history.json"
    if history_path.exists() and args.resume:
        i = 1
        while (exp_dir / "training" / f"history.json.{i}").exists():
            i += 1
        history_path.rename(exp_dir / "training" / f"history.json.{i}")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved to: {history_path}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print()
    print(f"Initial weights (θ₀): {initial_checkpoint_path}")
    print(f"Final model (θ*): {checkpoint_dir / 'final_model.pt'}")
    print(f"Training history: {history_path}")
    print()
    # Write DONE sentinel — this experiment is complete, never resume
    done_path = exp_dir / "training" / "DONE"
    if done:
        done_path.touch()

    print("Next steps:")
    print(f"  python scripts/evaluate.py --config {args.config}")
    print()

    # Save log — rotate old log on resume
    sys.stdout = _tee.original  # type: ignore[assignment]
    log_path = exp_dir / "training" / "train_maml.log"
    if log_path.exists() and args.resume:
        # Find next available suffix
        i = 1
        while (exp_dir / "training" / f"train_maml.log.{i}").exists():
            i += 1
        log_path.rename(exp_dir / "training" / f"train_maml.log.{i}")
    log_path.write_text(_tee.getvalue())
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
