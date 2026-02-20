"""
SVD and correlation analysis of PDE input features.

Usage:
    uv run scripts/dataset_analyser.py <directory> [--n_points 500] [--k_snapshots 5] [--seed 42]

Loads all *_fourier.npz files from the given directory, evaluates the 10
input features [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy] at
random spatial points across multiple timesteps, then runs SVD and correlation analysis.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fourier_eval
from src.data.fourier_eval import build_wavenumbers, fourier_eval_2d

FEATURE_NAMES = ["u", "v", "u_x", "u_y", "u_xx", "u_yy", "v_x", "v_y", "v_xx", "v_yy"]


def load_fourier_fields(npz_path: Path) -> dict:
    """Load a fourier .npz and return the two coefficient fields + metadata."""
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    if "A_hat" in keys:
        field1_hat = data["A_hat"]
        field2_hat = data["B_hat"]
    elif "u_hat" in keys:
        field1_hat = data["u_hat"]
        field2_hat = data["v_hat"]
    else:
        raise ValueError(f"Unknown .npz format in {npz_path}: keys = {keys}")

    sim_params = data["simulation_params"].item()
    times = data["times"]

    return {
        "field1_hat": field1_hat,
        "field2_hat": field2_hat,
        "sim_params": sim_params,
        "times": times,
    }


def evaluate_features(field1_hat, field2_hat, kx, ky, E_x, E_y):
    u = fourier_eval_2d(field1_hat, E_x, E_y)
    v = fourier_eval_2d(field2_hat, E_x, E_y)

    u_x = fourier_eval_2d(1j * kx.unsqueeze(0) * field1_hat, E_x, E_y)
    u_y = fourier_eval_2d(1j * ky.unsqueeze(1) * field1_hat, E_x, E_y)
    u_xx = fourier_eval_2d(((1j * kx.unsqueeze(0)) ** 2) * field1_hat, E_x, E_y)
    u_yy = fourier_eval_2d(((1j * ky.unsqueeze(1)) ** 2) * field1_hat, E_x, E_y)

    v_x = fourier_eval_2d(1j * kx.unsqueeze(0) * field2_hat, E_x, E_y)
    v_y = fourier_eval_2d(1j * ky.unsqueeze(1) * field2_hat, E_x, E_y)
    v_xx = fourier_eval_2d(((1j * kx.unsqueeze(0)) ** 2) * field2_hat, E_x, E_y)
    v_yy = fourier_eval_2d(((1j * ky.unsqueeze(1)) ** 2) * field2_hat, E_x, E_y)

    return torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

def build_phase_matrices(n_points, nx, ny, Lx, Ly, seed, device="cuda"):
    """Build phase matrices E_x, E_y for random spatial evaluation points."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x_pts = torch.rand(n_points, generator=gen, device=device, dtype=torch.float64) * Lx
    y_pts = torch.rand(n_points, generator=gen, device=device, dtype=torch.float64) * Ly

    kx, ky = build_wavenumbers(nx, ny, Lx, Ly, device=device)

    E_x = torch.exp(1j * x_pts.unsqueeze(1) * kx.unsqueeze(0))
    E_y = torch.exp(1j * y_pts.unsqueeze(1) * ky.unsqueeze(0))

    return kx, ky, E_x, E_y


def collect_features(directory: Path, n_points: int, seed: int, k_snapshots: int = 1) -> torch.Tensor:
    """Load all fourier .npz files and collect features into one big matrix.

    Args:
        directory: Path containing *_fourier.npz files
        n_points: Random spatial points per snapshot
        seed: Random seed for spatial point sampling
        k_snapshots: Number of evenly-spaced timesteps to sample per task
    """
    files = sorted(directory.glob("*_fourier.npz"))
    if not files:
        print(f"No *_fourier.npz files found in {directory}")
        sys.exit(1)

    print(f"Found {len(files)} fourier files in {directory}")
    print(f"Sampling {k_snapshots} timestep(s) per task, {n_points} spatial points each")

    all_features = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for f in files:
        loaded = load_fourier_fields(f)
        sim = loaded["sim_params"]

        domain_size = sim.get("domain_size", sim.get("Lx", 2 * np.pi))
        if isinstance(domain_size, (list, tuple)):
            Lx, Ly = domain_size
        else:
            Lx = Ly = float(domain_size)

        resolution = sim.get("resolution", sim.get("nx", 128))
        if isinstance(resolution, (list, tuple)):
            nx, ny = resolution
        else:
            nx = ny = int(resolution)

        n_total = loaded["field1_hat"].shape[0]
        indices = np.linspace(0, n_total - 1, k_snapshots, dtype=int)

        kx, ky, E_x, E_y = build_phase_matrices(n_points, nx, ny, Lx, Ly, seed, device)

        for idx in indices:
            f1 = torch.tensor(loaded["field1_hat"][idx], device=device)
            f2 = torch.tensor(loaded["field2_hat"][idx], device=device)

            feats = evaluate_features(f1, f2, kx, ky, E_x, E_y)
            all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)


def run_svd_analysis(features: torch.Tensor):
    """Run SVD on the standardized feature matrix and print results."""
    X = features.double()

    # Standardize: zero mean, unit variance per column
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    std[std == 0] = 1  # avoid division by zero for constant features
    X = (X - mean) / std

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    print("\n=== Singular Values ===")
    for i, s in enumerate(S):
        print(f"  sigma_{i+1:2d} = {s:.6f}")

    print(f"\n  Condition number: {S[0] / S[-1]:.2f}")

    cumulative = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    print("\n=== Cumulative Energy ===")
    for i, c in enumerate(cumulative):
        print(f"  {i+1:2d} components: {c:.4f}")

    print("\n=== Weakest Direction (v_10) ===")
    v_last = Vt[-1]
    for name, weight in zip(FEATURE_NAMES, v_last):
        print(f"  {name:>6s}: {weight:+.4f}")

    return S, Vt


def run_correlation_analysis(features: torch.Tensor):
    """Compute and return the Pearson correlation matrix."""
    corr = torch.corrcoef(features.T.double())

    print("\n=== Correlation Matrix ===")
    header = "        " + "  ".join(f"{n:>6s}" for n in FEATURE_NAMES)
    print(header)
    for i, name in enumerate(FEATURE_NAMES):
        row = f"  {name:>5s} " + "  ".join(f"{corr[i,j]:+.3f}" for j in range(10))
        print(row)

    return corr


def plot_results(S, Vt, corr, output_dir: Path):
    """Generate plots for singular values, weakest direction, and correlation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Singular value bar chart
    axes[0].bar(range(1, 11), S.numpy(), color="steelblue")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title("Singular Value Spectrum")
    axes[0].set_xticks(range(1, 11))

    # 2. Weakest direction weights
    v_last = Vt[-1].numpy()
    colors = ["tab:red" if abs(w) > 0.2 else "tab:gray" for w in v_last]
    axes[1].barh(range(10), v_last, color=colors)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels(FEATURE_NAMES)
    axes[1].set_xlabel("Weight")
    axes[1].set_title("Weakest Direction (v_10)")
    axes[1].axvline(x=0, color="black", linewidth=0.5)

    # 3. Correlation heatmap
    im = axes[2].imshow(corr.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(range(10))
    axes[2].set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    axes[2].set_yticks(range(10))
    axes[2].set_yticklabels(FEATURE_NAMES)
    axes[2].set_title("Pearson Correlation")
    fig.colorbar(im, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    out_path = output_dir / "svd_correlation_analysis.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SVD/correlation analysis of PDE features")
    parser.add_argument("directory", type=Path, help="Directory containing *_fourier.npz files")
    parser.add_argument("--n_points", type=int, default=500, help="Random spatial points per snapshot")
    parser.add_argument("--k_snapshots", type=int, default=5, help="Timesteps to sample per task (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=None, help="Where to save figures")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.directory

    features = collect_features(args.directory, args.n_points, args.seed, args.k_snapshots)
    print(f"\nFeature matrix shape: {features.shape}  (samples x features)")

    S, Vt = run_svd_analysis(features)
    corr = run_correlation_analysis(features)
    plot_results(S, Vt, corr, args.output_dir)


if __name__ == "__main__":
    main()
