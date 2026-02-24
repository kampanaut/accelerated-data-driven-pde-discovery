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

import yaml
import torch

from src.networks.pde_operator_network import PDEOperatorNetwork
from src.training.task_loader import MetaLearningDataLoader, TASK_REGISTRY
from src.training.maml import MAMLTrainer, MAMLConfig


def load_config(config_path: Path) -> dict:
    """Load and validate experiment configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required = ["experiment", "data", "training"]
    for section in required:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    return config


def setup_output_dirs(config: dict) -> Path:
    """Create experiment output directory structure."""
    base_dir = Path(config.get("output", {}).get("base_dir", "data/models"))
    exp_name = config["experiment"]["name"]
    exp_dir = base_dir / exp_name

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

    config = load_config(args.config)
    exp_dir = setup_output_dirs(config)

    # Tee stdout to capture all print output for log file
    _tee = _TeeStream(sys.stdout)
    sys.stdout = _tee  # type: ignore[assignment]

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Output directory: {exp_dir}")
    print()

    # Save config copy
    config_copy_path = exp_dir / "training" / "config.yaml"
    shutil.copy(args.config, config_copy_path)

    # =========================================================================
    # Set random seeds
    # =========================================================================
    seed = config["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config["experiment"].get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()

    # =========================================================================
    # Load datasets
    # =========================================================================
    print("-" * 60)
    print("Loading datasets...")
    print("-" * 60)

    train_dir = Path(config["data"]["meta_train_dir"])
    val_dir = Path(config["data"]["meta_val_dir"])

    if not train_dir.exists():
        raise FileNotFoundError(f"Meta-train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Meta-val directory not found: {val_dir}")

    # Select task class based on PDE type
    pde_type = config["experiment"].get("pde_type", "ns")
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
    val_loader = MetaLearningDataLoader(
        val_dir, task_class=task_class, task_pattern=task_pattern, device=device
    )

    print()
    print(f"Meta-train tasks: {len(train_loader)}")
    print(f"Meta-val tasks: {len(val_loader)}")
    print()

    # =========================================================================
    # Create model
    # =========================================================================
    print("-" * 60)
    print("Creating model...")
    print("-" * 60)

    train_cfg = config["training"]
    hidden_dims = train_cfg.get("hidden_dims", [100, 100])
    activation = train_cfg.get("activation", "tanh")

    input_dim = train_cfg.get("input_dim", 10)
    output_dim = train_cfg.get("output_dim", 2)

    model = PDEOperatorNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )
    print(model)
    print()

    # =========================================================================
    # Save initial weights (θ₀) for baseline comparison
    # =========================================================================
    initial_checkpoint_path = exp_dir / "checkpoints" / "initial_model.pt"

    if not args.resume:
        print("-" * 60)
        print("Saving initial weights (θ₀)...")
        print("-" * 60)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_dims": hidden_dims,
                    "activation": activation,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                },
                "timestamp": datetime.now().isoformat(),
            },
            initial_checkpoint_path,
        )

        print(f"Saved to: {initial_checkpoint_path}")
        print()
    else:
        print(f"Resume mode: using existing θ₀ from {initial_checkpoint_path}")
        print()

    # =========================================================================
    # Create MAML trainer
    # =========================================================================
    print("-" * 60)
    print("Configuring MAML trainer...")
    print("-" * 60)

    maml_config = MAMLConfig(
        inner_lr=train_cfg.get("inner_lr", 0.01),
        outer_lr=train_cfg.get("outer_lr", 0.001),
        inner_steps=train_cfg.get("inner_steps", 1),
        meta_batch_size=train_cfg.get("meta_batch_size", 4),
        k_shot=train_cfg.get("k_shot", 100),
        query_size=train_cfg.get("query_size", 1000),
        max_outer_iterations=train_cfg.get("max_iterations", 10000),
        patience=train_cfg.get("patience", 50),
        log_interval=train_cfg.get("log_interval", 10),
        first_order=train_cfg.get("first_order", False),
        warmup_iterations=train_cfg.get("warmup_iterations", 0),
        use_scheduler=train_cfg.get("use_scheduler", False),
        min_lr=train_cfg.get("min_lr", 1e-6),
        scheduler_type=train_cfg.get("scheduler_type", "cosine"),
        T_0=train_cfg.get("T_0", 500),
        T_mult=train_cfg.get("T_mult", 2),
        device=device,
        seed=seed,
    )

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
    history = trainer.train(checkpoint_dir=checkpoint_dir)

    # =========================================================================
    # Save training history
    # =========================================================================
    print()
    print("-" * 60)
    print("Saving training history...")
    print("-" * 60)

    history_path = exp_dir / "training" / "history.json"
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
    print(f"Best model (θ*): {checkpoint_dir / 'best_model.pt'}")
    print(f"Training history: {history_path}")
    print()
    print("Next steps:")
    print(f"  python scripts/evaluate.py --config {args.config}")
    print()

    # Save log
    sys.stdout = _tee.original  # type: ignore[assignment]
    log_path = exp_dir / "training" / "train_maml.log"
    log_path.write_text(_tee.getvalue())
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
