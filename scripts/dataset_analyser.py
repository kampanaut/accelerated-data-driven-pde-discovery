"""
SVD and R² analysis of PDE input features.

Usage:
    uv run scripts/dataset_analyser.py <directory> <pde_type> [--n_points 500] [--k_snapshots 5] [--seed 42]

Loads all *_fourier.npz files from the given directory, evaluates input
features at random spatial points across multiple timesteps, then runs
SVD, Pearson correlation, and R² redundancy analysis.

Outputs two figures:
  - feature_analysis[name].png  — 2x2: SVD spectrum, weakest direction,
                                   Pearson correlation heatmap, R²(feature|rest) bars
  - lap_analysis[name].png      — Laplacian-specific R² checks (2-field PDEs only)

Two-field PDEs (BR, NS, FHN, LO): 10 features [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]
Single-field PDEs (heat, nl_heat):  5 features [u, u_x, u_y, u_xx, u_yy]
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from typing import Any, ValuesView

import matplotlib.pyplot as plt

from src.training.task_loader import TASK_REGISTRY


class TeeStream:
    """Write to both stdout and a StringIO buffer."""

    def __init__(self, original: io.TextIOWrapper):  # type: ignore[type-arg]
        self.original = original
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.original.write(text)
        self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()  # type: ignore[union-attr]

    def getvalue(self) -> str:
        return self.buffer.getvalue()


FEATURE_NAMES_2F = [
    "u",
    "v",
    "u_x",
    "u_y",
    "u_xx",
    "u_yy",
    "v_x",
    "v_y",
    "v_xx",
    "v_yy",
]
FEATURE_NAMES_1F = ["u", "u_x", "u_y", "u_xx", "u_yy"]
TARGET_NAMES_2F = ["u_t", "v_t"]
TARGET_NAMES_1F = ["u_t"]


def collect_features(
    directory: Path,
    n_points: int,
    seed: int,
    k_snapshots: int = 1,
    pde_type: str = "br",
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Load all fourier .npz files and collect features + targets.

    Uses PDE task classes for evaluation, giving both input features and
    PDE RHS targets at random collocation points.

    Args:
        directory: Path containing *_fourier.npz files
        n_points: Random spatial points per snapshot
        seed: Random seed for spatial point sampling
        k_snapshots: Number of evenly-spaced timesteps to sample per task
        pde_type: PDE type key for TASK_REGISTRY

    Returns:
        (features, targets, feature_names, target_names)
    """
    files = sorted(directory.glob("*_fourier.npz"))
    if not files:
        print(f"No *_fourier.npz files found in {directory}")
        sys.exit(1)

    if pde_type not in TASK_REGISTRY:
        print(
            f"Unknown PDE type '{pde_type}'. Available: {list(TASK_REGISTRY.keys())}"
        )
        sys.exit(1)

    task_class = TASK_REGISTRY[pde_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Found {len(files)} fourier files in {directory}")
    print(f"PDE type: {pde_type} ({task_class.__name__})")
    print(
        f"Sampling {k_snapshots} timestep(s) per task, {n_points} spatial points each"
    )

    # Build phase matrices once using first task's grid
    first_task = task_class(files[0], device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x_pts = (
        torch.rand(n_points, generator=gen, device=device, dtype=torch.float64)
        * first_task.Lx
    )
    y_pts = (
        torch.rand(n_points, generator=gen, device=device, dtype=torch.float64)
        * first_task.Ly
    )
    E_x = torch.exp(
        1j * x_pts.unsqueeze(1) * first_task.kx.unsqueeze(0)
    )  # (n_points, nx)
    E_y = torch.exp(
        1j * y_pts.unsqueeze(1) * first_task.ky.unsqueeze(0)
    )  # (n_points, ny)
    E_x_batch = E_x.unsqueeze(0)  # (1, n_points, nx)
    E_y_batch = E_y.unsqueeze(0)  # (1, n_points, ny)

    n_feat = first_task.n_features
    n_tgt = first_task.n_targets
    feature_names = FEATURE_NAMES_2F if n_feat == 10 else FEATURE_NAMES_1F
    target_names = TARGET_NAMES_2F if n_tgt == 2 else TARGET_NAMES_1F

    all_features = []
    all_targets = []

    for f in files:
        task = task_class(f, device=device)
        indices = np.linspace(0, task.n_snapshots - 1, k_snapshots, dtype=int)

        for idx in indices:
            snap_idx = torch.tensor([int(idx)])
            feats, tgts = task.evaluate_collocations(
                snap_idx, E_x_batch, E_y_batch
            )
            # feats: (1, n_points, n_features), tgts: (1, n_points, n_targets)
            all_features.append(feats[0].cpu())
            all_targets.append(tgts[0].cpu())

    features = torch.cat(all_features, dim=0)
    targets = torch.cat(all_targets, dim=0)

    print(f"\nFeature matrix: {features.shape}  ({len(feature_names)} features)")
    print(f"Target matrix:  {targets.shape}  ({len(target_names)} targets)")

    return features, targets, feature_names, target_names


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def run_descriptive_stats(features: torch.Tensor, feature_names: list[str]):
    """Print per-feature descriptive statistics."""
    X = features.double()

    print("\n=== Descriptive Statistics (per feature) ===")
    header = f"  {'Feature':>6s}  {'min':>12s}  {'max':>12s}  {'mean':>12s}  {'median':>12s}  {'std':>12s}  {'var':>12s}  {'mean(x²)':>12s}  {'skew':>12s}  {'kurtosis':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, name in enumerate(feature_names):
        col = X[:, i]
        mu = col.mean()
        std = col.std()
        var = col.var()
        med = float(col.median())
        mn = float(col.min())
        mx = float(col.max())
        mean_sq = (col**2).mean()
        # Skewness: E[(x-μ)³] / σ³
        skew = ((col - mu) ** 3).mean() / (std**3 + 1e-30)
        # Excess kurtosis: E[(x-μ)⁴] / σ⁴ - 3
        kurt = ((col - mu) ** 4).mean() / (std**4 + 1e-30) - 3
        print(
            f"  {name:>6s}  {mn:>12.4f}  {mx:>12.4f}  {mu:>12.4f}  {med:>12.4f}"
            f"  {std:>12.4f}  {var:>12.4f}  {mean_sq:>12.4f}  {skew:>12.4f}  {kurt:>12.4f}"
        )

    # Overall
    flat = X.flatten()
    mu = flat.mean()
    std = flat.std()
    print(f"\n  Overall: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(
        f"  Global mean={mu:.6f}, std={std:.6f}, min={flat.min():.6f}, max={flat.max():.6f}"
    )
    print(f"  Global mean(x²)={(flat**2).mean():.6f}")


def r_squared(y: torch.Tensor, X: torch.Tensor) -> float:
    """OLS R² of regressing y on X (with intercept)."""
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    b = torch.linalg.lstsq(X_aug, y).solution
    y_pred = X_aug @ b
    ss_res = torch.sum((y - y_pred) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    return float(1 - (ss_res / ss_tot))


def run_svd_analysis(features: torch.Tensor, feature_names: list[str]):
    """Run SVD on the standardized feature matrix and print results."""
    n_feats = len(feature_names)
    X = features.double()

    # Standardize: zero mean, unit variance per column
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    std[std == 0] = 1  # avoid division by zero for constant features
    X = (X - mean) / std

    _U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    print("\n=== Singular Values ===")
    for i, s in enumerate(S):
        print(f"  sigma_{i + 1:2d} = {s:.6f}")

    print(f"\n  Condition number: {S[0] / S[-1]:.2f}")

    cumulative = torch.cumsum(S**2, dim=0) / (S**2).sum()
    print("\n=== Cumulative Energy ===")
    for i, c in enumerate(cumulative):
        print(f"  {i + 1:2d} components: {c:.4f}")

    print(f"\n=== Weakest Direction (v_{n_feats}) ===")
    v_last = Vt[-1]
    for name, weight in zip(feature_names, v_last):
        print(f"  {name:>6s}: {weight:+.4f}")

    return S, Vt


def run_correlation_analysis(features: torch.Tensor, feature_names: list[str]):
    """Compute and return the Pearson correlation matrix."""
    n_feats = len(feature_names)
    corr = torch.corrcoef(features.T.double())

    print("\n=== Correlation Matrix ===")
    header = "        " + "  ".join(f"{n:>6s}" for n in feature_names)
    print(header)
    for i, name in enumerate(feature_names):
        row = f"  {name:>5s} " + "  ".join(f"{corr[i, j]:+.3f}" for j in range(n_feats))
        print(row)

    return corr


def run_r2_analysis(features: torch.Tensor, feature_names: list[str]):
    """Compute R²(feature_i | all other features) for each feature."""
    n_feats = len(feature_names)
    r2_values = []

    print("\n=== R²(feature | rest) ===")
    for i in range(n_feats):
        y = features[:, i]
        others = torch.cat([features[:, :i], features[:, i + 1 :]], dim=1)
        val = r_squared(y, others)
        r2_values.append(val)
        print(f"  R²({feature_names[i]:>5s} | rest) = {val:.6f}")

    return r2_values


def run_lap_analysis(features: torch.Tensor, feature_names: list[str]):
    """Laplacian-specific R² analysis.

    Tests whether non-RHS feature groups can reconstruct the Laplacian(s).
    Supports both 2-field (10 features) and 1-field (5 features) PDEs.
    """
    n_feat = len(feature_names)

    if n_feat == 10:
        # 2-field: [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]
        #           0  1   2    3    4     5     6    7    8     9
        lap_u = features[:, 4] + features[:, 5]
        lap_v = features[:, 8] + features[:, 9]

        targets = {"∇²u": lap_u, "∇²v": lap_v}

        groups = {
            "[u, v]": [0, 1],
            "[u_x, u_y]": [2, 3],
            "[v_x, v_y]": [6, 7],
            "[u, v, u_x, u_y]": [0, 1, 2, 3],
            "[all \\ {u_xx,u_yy}]": [0, 1, 2, 3, 6, 7, 8, 9],
            "[all \\ {v_xx,v_yy}]": [0, 1, 2, 3, 4, 5, 6, 7],
            "[u_xx, u_yy]": [4, 5],
            "[v_xx, v_yy]": [8, 9],
            "[all]": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
    elif n_feat == 5:
        # 1-field: [u, u_x, u_y, u_xx, u_yy]
        #           0   1    2    3     4
        lap_u = features[:, 3] + features[:, 4]

        targets = {"∇²u": lap_u}

        groups = {
            "[u]": [0],
            "[u_x, u_y]": [1, 2],
            "[u, u_x, u_y]": [0, 1, 2],
            "[u_xx, u_yy]": [3, 4],
            "[all]": [0, 1, 2, 3, 4],
        }
    else:
        raise ValueError(f"Unsupported input features of {n_feat}")

    results: dict[str, dict[str, float]] = {}

    print("\n=== Laplacian R² Analysis ===")
    for tname, target in targets.items():
        results[tname] = {}
        # Per-feature pairwise r²
        print(f"\n  Pairwise r²({tname}, feature):")
        for i, fname in enumerate(feature_names):
            r = torch.corrcoef(torch.stack([target, features[:, i]]))[0, 1].item()
            results[tname][f"r²({fname})"] = r**2
            print(f"    r²({tname}, {fname:>5s}) = {r**2:.6f}")

        # Group R²
        print(f"\n  R²({tname} | group):")
        for gname, indices in groups.items():
            val = r_squared(target, features[:, indices])
            results[tname][gname] = val
            print(f"    R²({tname} | {gname:20s}) = {val:.6f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_feature_analysis(
    S,
    Vt,
    corr,
    r2_values: list[float],
    feature_names: list[str],
    corr_names: list[str],
    output_dir: Path,
):
    """2x2 figure: SVD spectrum, weakest direction, correlation heatmap, R² bars."""
    n_feats = len(feature_names)
    n_corr = len(corr_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Singular value bar chart
    axes[0, 0].bar(range(1, n_feats + 1), S.numpy(), color="steelblue")
    axes[0, 0].set_xlabel("Component")
    axes[0, 0].set_ylabel("Singular Value")
    axes[0, 0].set_title(f"Singular Value Spectrum  (κ = {S[0] / S[-1]:.2f})")
    axes[0, 0].set_xticks(range(1, n_feats + 1))

    # Top-right: Weakest direction weights
    v_last = Vt[-1].numpy()
    colors = ["tab:red" if abs(w) > 0.2 else "tab:gray" for w in v_last]
    axes[0, 1].barh(range(n_feats), v_last, color=colors)
    axes[0, 1].set_yticks(range(n_feats))
    axes[0, 1].set_yticklabels(feature_names)
    axes[0, 1].set_xlabel("Weight")
    axes[0, 1].set_title(f"Weakest Direction (v_{n_feats})")
    axes[0, 1].axvline(x=0, color="black", linewidth=0.5)

    # Bottom-left: Correlation heatmap (features + targets)
    im = axes[1, 0].imshow(corr.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(n_corr))
    axes[1, 0].set_xticklabels(corr_names, rotation=45, ha="right")
    axes[1, 0].set_yticks(range(n_corr))
    axes[1, 0].set_yticklabels(corr_names)
    axes[1, 0].set_title("Pearson Correlation (r)")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    # Bottom-right: R²(feature | rest) bar chart
    bar_colors = ["#d64545" if v > 0.5 else "#4a90d9" for v in r2_values]
    bars = axes[1, 1].bar(
        feature_names, r2_values, color=bar_colors, edgecolor="black", linewidth=0.5
    )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("R²")
    axes[1, 1].set_title("R²(feature | all others)  —  redundancy check")
    axes[1, 1].set_xticks(range(len(feature_names)))
    axes[1, 1].set_xticklabels(feature_names, rotation=45, ha="right")
    for bar, val in zip(bars, r2_values):
        axes[1, 1].text(
            bar.get_x() + (bar.get_width() / 2),
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = output_dir / f"feature_analysis[{output_dir.stem}].png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved to {out_path}")
    plt.close()


def _draw_lap_panel(
    ax: Any, labels: list[str], values: list[float], title: str
) -> None:
    """Draw a single bar chart panel for Laplacian R² analysis."""
    colors = ["#d64545" if v > 0.5 else "#4a90d9" for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, max(0.05, max(values) * 1.5))
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + (bar.get_width() / 2),
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )


def plot_lap_analysis(
    results: dict[str, dict[str, float]], feature_names: list[str], output_dir: Path
):
    """Laplacian-specific R² figure. Adapts layout to number of targets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = list(results.keys())
    n_rows = len(targets)
    groups = [k for k in results[targets[0]] if k.startswith(("[", "{"))]

    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing

    for row, tname in enumerate(targets):
        # Left: pairwise r²(target, each feature)
        pair_labels = list(feature_names)
        pair_vals = [results[tname][f"r²({fn})"] for fn in feature_names]
        _draw_lap_panel(
            axes[row, 0], pair_labels, pair_vals, f"r²({tname}, single feature)"
        )

        # Right: group R²(target | feature group)
        group_labels = list(groups)
        group_vals = [results[tname][gn] for gn in groups]
        _draw_lap_panel(
            axes[row, 1], group_labels, group_vals, f"R²({tname} | feature group)"
        )

    plt.tight_layout()
    out_path = output_dir / f"lap_analysis[{output_dir.stem}].png"
    plt.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SVD/R² analysis of PDE features")
    parser.add_argument(
        "directory", type=Path, help="Directory containing *_fourier.npz files"
    )
    parser.add_argument(
        "pde_type",
        type=str,
        choices=list(TASK_REGISTRY.keys()),
        help="PDE type key (br, fhn, lo, ns, heat, nl_heat)",
    )
    parser.add_argument(
        "--n_points", type=int, default=500, help="Random spatial points per snapshot"
    )
    parser.add_argument(
        "--k_snapshots",
        type=int,
        default=5,
        help="Timesteps to sample per task (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="Where to save figures"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.directory

    # Tee stdout to capture all print output for log file
    tee = TeeStream(sys.stdout)  # type: ignore[arg-type]
    sys.stdout = tee  # type: ignore[assignment]

    features, targets, feature_names, target_names = collect_features(
        args.directory, args.n_points, args.seed, args.k_snapshots, args.pde_type
    )

    # Descriptive stats for features + targets combined
    all_names = feature_names + target_names
    all_data = torch.cat([features, targets], dim=1)
    run_descriptive_stats(all_data, all_names)

    # SVD / R² on features only (input collinearity analysis)
    S, Vt = run_svd_analysis(features, feature_names)
    # Correlation on features + targets (shows input-output relationships)
    corr = run_correlation_analysis(all_data, all_names)
    r2_values = run_r2_analysis(features, feature_names)
    plot_feature_analysis(
        S, Vt, corr, r2_values, feature_names, all_names, args.output_dir
    )

    lap_results = run_lap_analysis(features, feature_names)
    plot_lap_analysis(lap_results, feature_names, args.output_dir)

    # Restore stdout and save log
    sys.stdout = tee.original
    log_path = args.output_dir / f"dataset_analyser[{args.output_dir.stem}].log"
    log_path.write_text(tee.getvalue())
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
