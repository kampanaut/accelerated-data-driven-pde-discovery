"""Inspect θ* weights from cheat (linear model) experiments.

For Heat: weights are [w_u, w_ux, w_uy, w_uxx, w_uyy].
The ideal per-task weights are [0, 0, 0, D, D].
θ* should encode the amortised starting point across the task distribution.

Usage:
    uv run python scripts/inspect_cheat_weights.py data/models/cheat/heat
    uv run python scripts/inspect_cheat_weights.py data/models/cheat/heat --include-initial
"""

import argparse
import sys
from pathlib import Path

import torch


FEATURE_NAMES = {
    5: ["u", "u_x", "u_y", "u_xx", "u_yy"],
    10: ["u", "v", "u_x", "u_y", "u_xx", "u_yy", "v_x", "v_y", "v_xx", "v_yy"],
}


def extract_weights(checkpoint_path: Path) -> torch.Tensor:
    """Load checkpoint and return the weight tensor."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    # Linear model: single key 'network.0.weight' with shape (output_dim, input_dim)
    weight_key = [k for k in state if "weight" in k]
    if len(weight_key) != 1:
        print(f"  WARNING: expected 1 weight tensor, found {len(weight_key)}: {weight_key}")
        return None
    return state[weight_key[0]]


def main():
    parser = argparse.ArgumentParser(description="Inspect cheat model weights")
    parser.add_argument("model_dir", type=Path, help="Directory containing experiment subdirs")
    parser.add_argument("--include-initial", action="store_true",
                        help="Also show θ₀ (random init) weights")
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Directory not found: {args.model_dir}")
        sys.exit(1)

    # Find all experiment dirs with best_model.pt
    exp_dirs = sorted([
        d for d in args.model_dir.iterdir()
        if d.is_dir() and (d / "checkpoints" / "best_model.pt").exists()
    ])

    if not exp_dirs:
        print(f"No trained models found in {args.model_dir}")
        sys.exit(1)

    print(f"Found {len(exp_dirs)} trained models\n")

    for exp_dir in exp_dirs:
        name = exp_dir.name
        best_path = exp_dir / "checkpoints" / "best_model.pt"
        weights = extract_weights(best_path)
        if weights is None:
            continue

        n_out, n_in = weights.shape
        labels = FEATURE_NAMES.get(n_in, [f"f{i}" for i in range(n_in)])

        # Print header
        print(f"{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")

        for out_idx in range(n_out):
            w = weights[out_idx]
            if n_out > 1:
                print(f"  output {out_idx}:")

            # Print each weight
            for i, (label, val) in enumerate(zip(labels, w)):
                marker = ""
                if abs(val.item()) < 0.01:
                    marker = "  ≈ 0"
                print(f"    w_{label:>5s} = {val.item():+.6f}{marker}")

            # For scalar PDEs: check if u_xx and u_yy weights match
            if n_in == 5:
                w_uxx, w_uyy = w[3].item(), w[4].item()
                avg_d = (w_uxx + w_uyy) / 2
                diff = abs(w_uxx - w_uyy)
                print(f"    ──────")
                print(f"    D_implied = (w_uxx + w_uyy)/2 = {avg_d:.6f}")
                print(f"    |w_uxx - w_uyy| = {diff:.6f}")

        if args.include_initial:
            init_path = exp_dir / "checkpoints" / "initial_model.pt"
            if init_path.exists():
                init_w = extract_weights(init_path)
                if init_w is not None:
                    print(f"    θ₀ weights: {init_w[0].tolist()}")

        print()


if __name__ == "__main__":
    main()
