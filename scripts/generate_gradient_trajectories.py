"""Generate gradient trajectories for visualization.

Loads a trained θ* checkpoint, picks one test task, and runs N fine-tuning
trajectories with different seeds. Records ∂L/∂w at every gradient step.

Output: (n_trajectories, n_steps, n_params) array saved to NPZ.

Usage:
    uv run python scripts/generate_gradient_trajectories.py \
        --config configs/cheat/heat-25-5step-k800-baseline.yaml \
        --task-idx 0 \
        --n-trajectories 5000 \
        --n-steps 1000 \
        --k-shot 5000 \
        --output data/gradient_trajectories.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.networks.pde_operator_network import NetworkConfig, PDEOperatorNetwork
from src.training.task_loader import TASK_REGISTRY, MetaLearningDataLoader


def run_trajectories(
    theta_star: PDEOperatorNetwork,
    tasks: list,
    n_trajectories: int,
    n_steps: int,
    k_shot: int,
    lr: float,
    device: str,
    base_seed: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Run multiple fine-tuning trajectories, recording gradients at every step.

    Each trajectory uniformly samples a task, samples K fresh collocation
    points from that task, then fine-tunes from θ* recording ∂L/∂w at
    every step.

    Returns:
        gradients: (n_trajectories, n_steps, n_params) array of per-weight gradients.
        task_indices: (n_trajectories,) list — which task each trajectory used.
    """
    n_params = sum(p.numel() for p in theta_star.parameters())
    all_gradients = np.zeros((n_trajectories, n_steps, n_params), dtype=np.float32)
    task_indices: list[int] = []
    rng = np.random.default_rng(base_seed)

    theta_star_state = theta_star.state_dict()

    for traj_idx in range(n_trajectories):
        if (traj_idx % 100 == 0) or (traj_idx == n_trajectories - 1):
            print(f"  Trajectory {traj_idx + 1}/{n_trajectories}", flush=True)

        # Uniformly sample a task
        task_idx = int(rng.integers(0, len(tasks)))
        task_indices.append(task_idx)
        task = tasks[task_idx]

        # Sample K fresh collocation points (unique per trajectory)
        (x, y), _, _, _ = task.get_support_query_split(
            K_shot=k_shot, query_size=0, k_seed=base_seed + traj_idx,
        )

        # Fresh copy of θ*
        model = PDEOperatorNetwork(theta_star.config)
        model.load_state_dict(theta_star_state)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()

            # Record gradients before optimizer step
            grad_flat = torch.cat([
                p.grad.detach().flatten() for p in model.parameters()
            ]).cpu().numpy()
            all_gradients[traj_idx, step] = grad_flat

            optimizer.step()

    return all_gradients, task_indices


def main():
    parser = argparse.ArgumentParser(description="Generate gradient trajectories")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-trajectories", type=int, default=5000)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--k-shot", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Fine-tuning learning rate")
    parser.add_argument("--output", type=Path, default=Path("data/gradient_trajectories.npz"))
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config["experiment"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load model
    train_cfg = config["training"]
    net_config = NetworkConfig.from_dict(train_cfg)

    exp_name = config["experiment"]["name"]
    base_dir = Path(config["output"]["base_dir"])
    exp_dir = base_dir / exp_name
    theta_star_path = exp_dir / "checkpoints" / "best_model.pt"

    if not theta_star_path.exists():
        print(f"Checkpoint not found: {theta_star_path}")
        sys.exit(1)

    checkpoint = torch.load(theta_star_path, map_location=device, weights_only=False)
    theta_star = PDEOperatorNetwork(net_config)
    theta_star.load_state_dict(checkpoint["model_state_dict"])
    theta_star = theta_star.to(device)
    print(f"Loaded θ* from {theta_star_path}")
    print(theta_star)

    # Load test tasks
    pde_type = config["experiment"]["pde_type"]
    test_dir = config["data"]["meta_test_dir"]
    pde_class = TASK_REGISTRY[pde_type]
    test_loader = MetaLearningDataLoader(test_dir, pde_class)

    tasks = test_loader.tasks
    task_names = [t.task_name for t in tasks]
    print(f"Test tasks: {len(tasks)}")
    for i, t in enumerate(tasks):
        specs = t.coefficient_specs
        coeff_str = ", ".join(f"{s.name}={s.true_value:.4f}" for s in specs)
        print(f"  [{i}] {t.task_name}: {coeff_str}")

    n_params = sum(p.numel() for p in theta_star.parameters())
    print(f"Parameters: {n_params}")
    print()

    # Run trajectories (uniformly sampling across all tasks)
    print(f"Running {args.n_trajectories} trajectories × {args.n_steps} steps "
          f"(K={args.k_shot}, lr={args.lr}, tasks={len(tasks)})...")
    gradients, task_indices = run_trajectories(
        theta_star, tasks,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        k_shot=args.k_shot,
        lr=args.lr,
        device=device,
    )

    # Collect per-task coefficient info
    coeff_values = {}
    for t in tasks:
        for s in t.coefficient_specs:
            coeff_values[f"{t.task_name}/{s.name}"] = s.true_value

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        gradients=gradients,  # (n_trajectories, n_steps, n_params)
        task_indices=np.array(task_indices),  # (n_trajectories,)
        task_names=np.array(task_names),  # (n_tasks,)
        config_path=str(args.config),
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        k_shot=args.k_shot,
        lr=args.lr,
    )
    print(f"\nSaved: {args.output}")
    print(f"Shape: {gradients.shape}")
    size_mb = gradients.nbytes / (1024 * 1024)
    print(f"Size: {size_mb:.1f} MB")

    # Task distribution summary
    task_idx_arr = np.array(task_indices)
    for i, name in enumerate(task_names):
        count = (task_idx_arr == i).sum()
        print(f"  {name}: {count} trajectories")


if __name__ == "__main__":
    main()
