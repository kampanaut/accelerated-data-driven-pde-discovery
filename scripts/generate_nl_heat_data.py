"""
Generate nonlinear heat equation training data for PDE discovery.

Solves u_t = K*(1-u)*nabla^2(u) for each IC, FFTs snapshots to Fourier
coefficients, saves .npz files.

Usage:
    python scripts/generate_nl_heat_data.py --config configs/nl_heat_train-1.yaml
    python scripts/generate_nl_heat_data.py --config configs/nl_heat_train-1.yaml --workers 4
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.data.generation import PDESpec, FieldType, run_generation
from src.data.initial_conditions_heat import create_heat_ic
from src.pde.nl_heat_equation import solve_nl_heat


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_nl_heat_evolution(
    field_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    n_snapshots: int = 8,
) -> None:
    """
    Save a multi-panel figure showing nonlinear heat equation evolution over time.

    Creates a 3xn_snapshots grid:
    - Row 1: u field (2D heatmap)
    - Row 2: u field (3D surface)
    - Row 3: |nabla u| gradient magnitude (2D heatmap)
    """
    indices = np.linspace(0, len(field_history) - 1, n_snapshots, dtype=int)

    X, Y = np.meshgrid(x, y)
    Lx, Ly = float(x[-1]), float(y[-1])
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    u_sel = [field_history[i] for i in indices]

    grad_sel = []
    for u_snap in u_sel:
        uy, ux = np.gradient(u_snap, dy, dx)
        grad_sel.append(np.sqrt(ux**2 + uy**2))

    u_min, u_max = min(a.min() for a in u_sel), max(a.max() for a in u_sel)
    grad_min, grad_max = min(a.min() for a in grad_sel), max(a.max() for a in grad_sel)

    fig = plt.figure(figsize=(4 * n_snapshots, 14))

    for col, idx in enumerate(indices):
        t = times[idx]

        # Row 1: u field (2D)
        ax = plt.subplot(3, n_snapshots, col + 1)
        im = ax.imshow(
            u_sel[col],
            cmap="inferno",
            vmin=u_min,
            vmax=u_max,
            origin="lower",
            extent=(0, Lx, 0, Ly),
        )
        ax.set_title(f"t={t:.2f}", fontsize=10)
        if col == 0:
            ax.set_ylabel("u", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 2: u 3D surface (downsampled)
        ax3d = fig.add_subplot(
            3,
            n_snapshots,
            n_snapshots + col + 1,
            projection="3d",
        )
        assert isinstance(ax3d, Axes3D)
        step = max(1, X.shape[0] // 32)
        ax3d.plot_surface(
            X[::step, ::step],
            Y[::step, ::step],
            u_sel[col][::step, ::step],
            cmap="inferno",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        ax3d.set_xlabel("x", fontsize=8)
        ax3d.set_ylabel("y", fontsize=8)
        ax3d.set_zlabel("u", fontsize=8)
        ax3d.view_init(elev=30, azim=45)
        ax3d.tick_params(labelsize=6)

        # Row 3: |nabla u| gradient magnitude
        ax = plt.subplot(3, n_snapshots, 2 * n_snapshots + col + 1)
        im = ax.imshow(
            grad_sel[col],
            cmap="magma",
            vmin=grad_min,
            vmax=grad_max,
            origin="lower",
            extent=(0, Lx, 0, Ly),
        )
        if col == 0:
            ax.set_ylabel("|∇u|", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# PDESpec
# ---------------------------------------------------------------------------

nl_heat_spec = PDESpec(
    name="Nonlinear Heat Equation",
    pde_param_keys=["K"],
    create_ic=create_heat_ic,
    solve=solve_nl_heat,
    field_type=FieldType.SCALAR,
    default_output_name="nl_heat",
    extra_save_keys=[("K_used", "K")],
    save_visualization=save_nl_heat_evolution,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate nonlinear heat equation training data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    run_generation(nl_heat_spec, args.config, args.workers)
