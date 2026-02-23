"""
Generate Lambda-Omega training data for PDE discovery.

Solves Lambda-Omega reaction-diffusion equations for each IC, FFTs
snapshots to Fourier coefficients, saves .npz files.

Usage:
    python scripts/generate_lo_data.py --config configs/lo_train-1.yaml
    python scripts/generate_lo_data.py --config configs/lo_train-1.yaml --workers 4
"""

import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generation import PDESpec, FieldType, run_generation
from src.data.initial_conditions_lo import create_lo_ic
from src.pde.lambda_omega import solve_lo


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def prepare_lo_ic_config(ic_config: dict, sim_params: dict) -> dict:
    """Inject amplitude parameter 'a' into IC config for limit-cycle scaling."""
    ic_config = ic_config.copy()
    ic_config["a_value"] = sim_params["a"]
    return ic_config


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_lo_evolution(
    field_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    n_snapshots: int = 8,
) -> None:
    """
    Save a multi-panel figure showing Lambda-Omega evolution over time.

    Creates a 5xn_snapshots grid:
    - Row 1: u field (2D heatmap)
    - Row 2: v field (2D heatmap)
    - Row 3: phase theta = atan2(v, u)
    - Row 4: amplitude r = sqrt(u^2 + v^2)
    - Row 5: u field (3D surface)
    """
    indices = np.linspace(0, len(field_history) - 1, n_snapshots, dtype=int)

    X, Y = np.meshgrid(x, y)
    Lx, Ly = float(x[-1]), float(y[-1])

    u_sel = [field_history[i][0] for i in indices]
    v_sel = [field_history[i][1] for i in indices]
    phase_sel = [np.arctan2(v, u) for u, v in zip(u_sel, v_sel)]
    amp_sel = [np.sqrt(u**2 + v**2) for u, v in zip(u_sel, v_sel)]

    u_min, u_max = min(a.min() for a in u_sel), max(a.max() for a in u_sel)
    v_min, v_max = min(a.min() for a in v_sel), max(a.max() for a in v_sel)
    amp_min, amp_max = min(a.min() for a in amp_sel), max(a.max() for a in amp_sel)

    fig = plt.figure(figsize=(4 * n_snapshots, 22))

    for col, idx in enumerate(indices):
        t = times[idx]

        # Row 1: u field
        ax = plt.subplot(5, n_snapshots, col + 1)
        im = ax.imshow(
            u_sel[col],
            cmap="RdBu_r",
            vmin=u_min,
            vmax=u_max,
            origin="lower",
            extent=(0, Lx, 0, Ly),
        )
        ax.set_title(f"t={t:.1f}", fontsize=10)
        if col == 0:
            ax.set_ylabel("u", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 2: v field
        ax = plt.subplot(5, n_snapshots, n_snapshots + col + 1)
        im = ax.imshow(
            v_sel[col],
            cmap="RdBu_r",
            vmin=v_min,
            vmax=v_max,
            origin="lower",
            extent=(0, Lx, 0, Ly),
        )
        if col == 0:
            ax.set_ylabel("v", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 3: phase
        ax = plt.subplot(5, n_snapshots, 2 * n_snapshots + col + 1)
        im = ax.imshow(
            phase_sel[col],
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            origin="lower",
            extent=(0, Lx, 0, Ly),
        )
        if col == 0:
            ax.set_ylabel("θ = atan2(v,u)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 4: amplitude
        ax = plt.subplot(5, n_snapshots, 3 * n_snapshots + col + 1)
        im = ax.imshow(
            amp_sel[col],
            cmap="viridis",
            vmin=amp_min,
            vmax=amp_max,
            origin="lower",
            extent=(0, Lx, 0, Ly),
        )
        if col == 0:
            ax.set_ylabel("r = √(u²+v²)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 5: u 3D surface (downsampled)
        ax3d = fig.add_subplot(
            5,
            n_snapshots,
            4 * n_snapshots + col + 1,
            projection="3d",
        )
        assert isinstance(ax3d, Axes3D)
        step = max(1, X.shape[0] // 32)
        ax3d.plot_surface(
            X[::step, ::step],
            Y[::step, ::step],
            u_sel[col][::step, ::step],
            cmap="RdBu_r",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        ax3d.set_xlabel("x", fontsize=8)
        ax3d.set_ylabel("y", fontsize=8)
        ax3d.set_zlabel("u", fontsize=8)
        ax3d.view_init(elev=30, azim=45)
        ax3d.tick_params(labelsize=6)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# PDESpec
# ---------------------------------------------------------------------------

lo_spec = PDESpec(
    name="Lambda-Omega",
    pde_param_keys=["D_u", "D_v", "a", "c"],
    samplable_params=["D_u", "D_v", "c"],
    create_ic=create_lo_ic,
    solve=solve_lo,
    field_type=FieldType.PAIRED,
    default_output_name="lo",
    prepare_ic_config=prepare_lo_ic_config,
    save_visualization=save_lo_evolution,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Lambda-Omega training data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    run_generation(lo_spec, args.config, args.workers)
