"""
Parameter difference sanity check for meta-learned models.

Usage:
    uv run scripts/parameter_sanity_check.py <initial_dir> <final_dir> [--with-gradient <dataset_dir>]

Compares two model checkpoints and reports whether training moved
parameters meaningfully. Implements Chris's suggestion (2026-02-10):
compute ∇L(θ) · (θ' - θ) and check if it's greater than a threshold.

Thresholds:
    |∇L · Δθ| < 1e-6: not meaningful (floating point noise)
    |∇L · Δθ| ~ 1e-3: weak
    |∇L · Δθ| ~ 1e-1: meaningful
    |∇L · Δθ| > 1e-1: strong
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.task_loader import BrusselatorTask, FitzHughNagumoTask, LambdaOmegaTask, MetaLearningDataLoader, NavierStokesTask
from src.networks.pde_operator_network import PDEOperatorNetwork


def load_checkpoint(path: Path) -> dict:
    """Load a single .pt checkpoint file."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def state_to_vec(state_dict: dict) -> torch.Tensor:
    """Flatten all parameters in a state dict into a single 1D vector."""
    return torch.cat([p.flatten().float() for p in state_dict.values()])


def compute_parameter_norms(theta_0: dict, theta_star: dict):
    """Compute parameter difference norms and per-layer breakdown."""
    print("\n=== Parameter Difference (Fast Check) ===\n")

    vec_0 = state_to_vec(theta_0)
    vec_star = state_to_vec(theta_star)
    diff = vec_star - vec_0

    l2 = torch.norm(diff).item()
    l1 = torch.norm(diff, p=1).item()
    linf = torch.norm(diff, p=float("inf")).item()
    cosine = torch.nn.functional.cosine_similarity(vec_0.unsqueeze(0), vec_star.unsqueeze(0)).item()

    print(f"  ||θ' - θ||₂ (L2 norm):    {l2:.6e}")
    print(f"  ||θ' - θ||₁ (L1 norm):    {l1:.6e}")
    print(f"  ||θ' - θ||∞ (max abs):    {linf:.6e}")
    print(f"  mean(|θ' - θ|):           {diff.abs().mean().item():.6e}")
    print(f"  cos(θ, θ'):               {cosine:.6f}")

    print("\n  Per-layer breakdown:")
    for name in theta_0:
        p0 = theta_0[name].float()
        ps = theta_star[name].float()
        layer_diff = (ps - p0).norm().item()
        layer_norm = p0.norm().item()
        relative = (layer_diff / layer_norm * 100) if layer_norm > 0 else float("inf")
        print(f"    {name:>25s}:  ||Δ||={layer_diff:.4e}  ({relative:.2f}% relative)")

    # Interpret
    print("\n  Interpretation:")
    if l2 < 1e-6:
        print("    ✗ No meaningful change — parameters barely moved (floating point noise)")
    elif l2 < 1e-3:
        print("    ~ Weak change — parameters moved slightly")
    elif l2 < 1e-1:
        print("    ✓ Moderate change — parameters moved meaningfully")
    else:
        print("    ✓ Strong change — parameters moved significantly")


def compute_gradient_check(theta_0, theta_star, config, dataset_dir, pde_type):
    model = PDEOperatorNetwork(
        input_dim=config.get("input_dim", 10),
        output_dim=config.get("output_dim", 2),
        hidden_dims=config.get("hidden_dims", [100, 100]),
        activation=config.get("activation", "tanh"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model.load_state_dict(theta_0)

    if pde_type not in ["br", "fhn", "lo", "ns"]:
        raise ValueError("pde_type is invalid")

    task_classes = {
            "br": BrusselatorTask, 
            "ns": NavierStokesTask,
            "fhn": FitzHughNagumoTask,
            "lo": LambdaOmegaTask
    }

    pde_class = task_classes[pde_type]

    dataset_loader = MetaLearningDataLoader(dataset_dir, pde_class)

    all_X, all_y = [], []
    for _, task in enumerate(dataset_loader.tasks):
        (X,y), _ = task.get_support_query_split(K_shot=30, query_size=0)
        all_X.append(X)
        all_y.append(y)

    X = torch.cat(all_X)
    y = torch.cat(all_y)

    pred_y = model(X)
    loss = F.mse_loss(pred_y, y)
    grad_vec = torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))])
    diff_vec = (state_to_vec(theta_star) - state_to_vec(theta_0)).to(device)

    dot_product = torch.dot(grad_vec, diff_vec).item()
    cosine = F.cosine_similarity(grad_vec.unsqueeze(0), diff_vec.unsqueeze(0)).item()
    grad_norm = torch.norm(grad_vec).item()
    diff_norm = torch.norm(diff_vec).item()

    print("\n=== Gradient Check ===\n")
    print(f"  Data: {len(all_X)} tasks x 30 points = {X.shape[0]} samples from {dataset_dir}")
    print(f"  Loss at θ₀:              {loss.item():.6e}")
    print(f"  ||∇L(θ₀)||₂:            {grad_norm:.6e}")
    print(f"  ||θ' - θ₀||₂:           {diff_norm:.6e}")
    print(f"  ∇L(θ₀) · (θ' - θ₀):    {dot_product:.6e}")
    print(f"  cos(∇L, θ' - θ₀):       {cosine:.6f}")

    print("\n  Interpretation:")
    abs_dot = abs(dot_product)
    if abs_dot < 1e-6:
        print("    ✗ Not meaningful (floating point noise)")
    elif abs_dot < 1e-3:
        print("    ~ Weak effect")
    elif abs_dot < 1e-1:
        print("    ✓ Meaningful")
    else:
        print("    ✓ Strong")

    if dot_product < 0:
        print("    ↓ Negative: moved downhill (training reduced loss)")
    elif dot_product > 0:
        print("    ↑ Positive: moved uphill (suspicious)")
    else:
        print("    → Zero: moved perpendicular to gradient")

    if abs(cosine) < 0.1:
        print("    ⊥ Nearly orthogonal: training direction ≠ gradient direction")
    elif cosine < -0.5:
        print("    ↓ Strong alignment with descent direction")
    elif cosine > 0.5:
        print("    ↑ Strong alignment with ascent direction (suspicious)")










def main():
    parser = argparse.ArgumentParser(description="Parameter difference sanity check")
    parser.add_argument("theta_0_path", type=Path, help="Path to initial .pt checkpoint")
    parser.add_argument("theta_star_path", type=Path, help="Path to final .pt checkpoint")
    parser.add_argument("--with-gradient", type=Path, default=None,
                        help="Dataset directory used to train θ₀ → θ' (for gradient check)")
    parser.add_argument("--pde-type", type=str, default=None, choices=["ns", "br", "fhn", "lo"],
                        help="PDE type of the dataset (required with --with-gradient)")
    args = parser.parse_args()

    if args.with_gradient is not None and args.pde_type is None:
        parser.error("--pde-type is required when using --with-gradient")

    print(f"Comparing:\n  θ₀:  {args.theta_0_path}\n  θ':  {args.theta_star_path}")

    ckpt_0 = load_checkpoint(args.theta_0_path)
    ckpt_star = load_checkpoint(args.theta_star_path)

    theta_0 = ckpt_0["model_state_dict"]
    theta_star = ckpt_star["model_state_dict"]
    config = ckpt_0.get("config", {})

    total_params = sum(p.numel() for p in theta_0.values())
    print(f"Total parameters: {total_params:,}")

    compute_parameter_norms(theta_0, theta_star)

    if args.with_gradient is not None:
        compute_gradient_check(theta_0, theta_star, config, args.with_gradient, args.pde_type)
    else:
        print("\n  (Skipping gradient check — pass --with-gradient <dataset_dir> --pde-type <ns|br|fhn|lo> to enable)")


if __name__ == "__main__":
    main()
