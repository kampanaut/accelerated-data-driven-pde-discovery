"""Test JVP coefficient extraction with different tangent scalings.

Loads a cheat model (linear, 5 params) and compares:
1. Current method: tangent [0, 0, 0, 1, 1], divide by 2
2. Proposed method: tangent [0, 0, 0, 0.5, 0.5], no division
3. Direct weight read (ground truth for linear model)

Usage:
    uv run python scripts/test_jvp_tangent.py configs/cheat/heat-17-1step-k800-baseline.yaml
"""

import sys
from pathlib import Path

import torch
import numpy as np
import yaml
from torch.autograd.functional import jvp

from src.networks.pde_operator_network import NetworkConfig, PDEOperatorNetwork
from src.training.task_loader import TASK_REGISTRY, MetaLearningDataLoader


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_jvp_tangent.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    train_cfg = config["training"]
    net_config = NetworkConfig.from_dict(train_cfg)

    exp_name = config["experiment"]["name"]
    base_dir = Path(config["output"]["base_dir"])
    exp_dir = base_dir / exp_name
    theta_star_path = exp_dir / "checkpoints" / "best_model.pt"

    checkpoint = torch.load(theta_star_path, map_location=device, weights_only=False)
    model = PDEOperatorNetwork(net_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Print actual weights
    weights = list(model.parameters())[0].detach().cpu().numpy().flatten()
    print("=" * 60)
    print("Direct weight read (ground truth for linear model)")
    print("=" * 60)
    labels = ["u", "u_x", "u_y", "u_xx", "u_yy"]
    for label, w in zip(labels, weights):
        print(f"  w_{label:>5s} = {w:+.6f}")
    print(f"  D_true_from_weights = (w_uxx + w_uyy) / 2 = {(weights[3] + weights[4]) / 2:.6f}")
    print()

    # Load a test task for collocation points
    pde_type = config["experiment"]["pde_type"]
    test_dir = config["data"]["meta_test_dir"]
    pde_class = TASK_REGISTRY[pde_type]
    loader = MetaLearningDataLoader(test_dir, pde_class, device=device)
    task = loader.tasks[0]

    print(f"\nTask: {task.task_name}, true D = {task.D:.6f}")
    print()

    # Sample collocation points
    (features, targets), _, _, _ = task.get_support_query_split(
        K_shot=1000, query_size=0, k_seed=42
    )

    def forward(x):
        return model(x)

    # Method 1: tangent [0, 0, 0, 1, 1], divide by len(perturb_indices)
    tangent_1 = torch.zeros_like(features)
    tangent_1[:, 3] = 1.0
    tangent_1[:, 4] = 1.0

    _, jvp_1 = jvp(forward, (features,), (tangent_1,))
    coeff_1 = (jvp_1[:, 0] / 2).detach().cpu().numpy()

    print("=" * 60)
    print("Method 1: tangent [0, 0, 0, 1, 1] / 2")
    print("=" * 60)
    print(f"  Mean D = {np.mean(coeff_1):.6f}")
    print(f"  Std  D = {np.std(coeff_1):.6f}")
    print(f"  Min  D = {np.min(coeff_1):.6f}")
    print(f"  Max  D = {np.max(coeff_1):.6f}")
    print()

    # Method 2: tangent [0, 0, 0, 0.5, 0.5], no division
    tangent_2 = torch.zeros_like(features)
    tangent_2[:, 3] = 0.5
    tangent_2[:, 4] = 0.5

    _, jvp_2 = jvp(forward, (features,), (tangent_2,))
    coeff_2 = jvp_2[:, 0].detach().cpu().numpy()

    print("=" * 60)
    print("Method 2: tangent [0, 0, 0, 0.5, 0.5] (no division)")
    print("=" * 60)
    print(f"  Mean D = {np.mean(coeff_2):.6f}")
    print(f"  Std  D = {np.std(coeff_2):.6f}")
    print(f"  Min  D = {np.min(coeff_2):.6f}")
    print(f"  Max  D = {np.max(coeff_2):.6f}")
    print()

    # Method 3: separate JVPs for u_xx and u_yy
    tangent_xx = torch.zeros_like(features)
    tangent_xx[:, 3] = 1.0

    tangent_yy = torch.zeros_like(features)
    tangent_yy[:, 4] = 1.0

    _, jvp_xx = jvp(forward, (features,), (tangent_xx,))
    _, jvp_yy = jvp(forward, (features,), (tangent_yy,))

    coeff_xx = jvp_xx[:, 0].detach().cpu().numpy()
    coeff_yy = jvp_yy[:, 0].detach().cpu().numpy()

    print("=" * 60)
    print("Method 3: separate JVPs (∂u_t/∂u_xx and ∂u_t/∂u_yy)")
    print("=" * 60)
    print(f"  ∂u_t/∂u_xx: mean = {np.mean(coeff_xx):.6f}, std = {np.std(coeff_xx):.6f}")
    print(f"  ∂u_t/∂u_yy: mean = {np.mean(coeff_yy):.6f}, std = {np.std(coeff_yy):.6f}")
    print(f"  Average:     mean = {(np.mean(coeff_xx) + np.mean(coeff_yy)) / 2:.6f}")
    print()

    # Comparison
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  Method 1 == Method 2?  max|diff| = {np.max(np.abs(coeff_1 - coeff_2)):.2e}")
    print(f"  w_uxx (direct)       = {weights[3]:+.6f}")
    print(f"  w_uyy (direct)       = {weights[4]:+.6f}")
    print(f"  ∂u_t/∂u_xx (JVP)    = {np.mean(coeff_xx):+.6f}")
    print(f"  ∂u_t/∂u_yy (JVP)    = {np.mean(coeff_yy):+.6f}")
    print(f"  JVP matches weights? |w_uxx - JVP_xx| = {abs(weights[3] - np.mean(coeff_xx)):.2e}")
    print(f"                       |w_uyy - JVP_yy| = {abs(weights[4] - np.mean(coeff_yy)):.2e}")


if __name__ == "__main__":
    main()
