"""
Generate Brusselator training data for PDE discovery.

Solves Brusselator reaction-diffusion equations for each IC, FFTs
snapshots to Fourier coefficients, saves .npz files.

Usage:
    python scripts/generate_br_data.py --config configs/br_train-1.yaml
    python scripts/generate_br_data.py --config configs/br_train-1.yaml --workers 4
"""

import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generation import PDESpec, FieldType, run_generation
from src.data.initial_conditions_brusselator import (
    create_brusselator_ic,
    compute_turing_threshold,
)
from src.pde.brusselator import solve_br


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def prepare_br_ic_config(ic_config: dict, sim_params: dict) -> dict:
    """Inject k1/k2 into IC config for steady-state calculation."""
    ic_config = ic_config.copy()
    ic_config["k1"] = sim_params["k1"]
    ic_config["k2"] = sim_params["k2"]
    return ic_config


def post_sample_br_params(
    ic_config: dict, sim_params: dict, rng: np.random.Generator
) -> dict:
    """Handle k2_delta mode: sample k2 relative to Turing threshold."""
    k2_delta_range = ic_config.get("k2_delta")
    if k2_delta_range is not None:
        k2_c = compute_turing_threshold(
            k1=sim_params["k1"],
            D_u=sim_params["D_u"],
            D_v=sim_params["D_v"],
        )
        delta = rng.uniform(k2_delta_range[0], k2_delta_range[1])
        sim_params = sim_params.copy()
        sim_params["k2"] = k2_c + delta
    return sim_params


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_brusselator_evolution(
    concentration_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    n_snapshots: int = 8,
) -> None:
    """
    Save a multi-panel figure showing Brusselator evolution over time.

    Creates a 4xn_snapshots grid:
    - Row 1: u concentration (2D contour)
    - Row 2: v concentration (2D contour)
    - Row 3: u concentration (3D surface)
    - Row 4: v concentration (3D surface)
    """
    import matplotlib.pyplot as plt

    indices = np.linspace(0, len(concentration_history) - 1, n_snapshots, dtype=int)

    fig = plt.figure(figsize=(5 * n_snapshots, 20))

    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y)
    else:
        x_grid, y_grid = x, y

    all_u = [concentration_history[i][0] for i in indices]
    all_v = [concentration_history[i][1] for i in indices]
    u_min, u_max = min(a.min() for a in all_u), max(a.max() for a in all_u)
    v_min, v_max = min(b.min() for b in all_v), max(b.max() for b in all_v)

    for col, idx in enumerate(indices):
        u, v = concentration_history[idx]
        t = times[idx]

        # Row 1: u concentration (2D)
        ax_u_2d = plt.subplot(4, n_snapshots, col + 1)
        contour_u = ax_u_2d.contourf(
            x_grid, y_grid, u, levels=20, cmap="YlOrRd", vmin=u_min, vmax=u_max
        )
        plt.colorbar(contour_u, ax=ax_u_2d, fraction=0.046, pad=0.04)
        ax_u_2d.set_title(f"t={t:.3f}")
        ax_u_2d.set_aspect("equal")

        # Row 2: v concentration (2D)
        ax_v_2d = plt.subplot(4, n_snapshots, n_snapshots + col + 1)
        contour_v = ax_v_2d.contourf(
            x_grid, y_grid, v, levels=20, cmap="YlGnBu", vmin=v_min, vmax=v_max
        )
        plt.colorbar(contour_v, ax=ax_v_2d, fraction=0.046, pad=0.04)
        ax_v_2d.set_aspect("equal")

        # Row 3: u concentration (3D surface)
        ax_u_3d = plt.subplot(
            4, n_snapshots, 2 * n_snapshots + col + 1, projection="3d"
        )
        assert isinstance(ax_u_3d, Axes3D)
        step = max(1, x_grid.shape[0] // 32)
        surf_u = ax_u_3d.plot_surface(
            x_grid[::step, ::step], y_grid[::step, ::step], u[::step, ::step],
            cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.9,
        )
        ax_u_3d.set_xlabel("x")
        ax_u_3d.set_ylabel("y")
        ax_u_3d.set_zlabel("u")
        ax_u_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_u, ax=ax_u_3d, fraction=0.03, pad=0.1, shrink=0.5)

        # Row 4: v concentration (3D surface)
        ax_v_3d = plt.subplot(
            4, n_snapshots, 3 * n_snapshots + col + 1, projection="3d"
        )
        assert isinstance(ax_v_3d, Axes3D)
        surf_v = ax_v_3d.plot_surface(
            x_grid[::step, ::step], y_grid[::step, ::step], v[::step, ::step],
            cmap="YlGnBu", linewidth=0, antialiased=True, alpha=0.9,
        )
        ax_v_3d.set_xlabel("x")
        ax_v_3d.set_ylabel("y")
        ax_v_3d.set_zlabel("v")
        ax_v_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_v, ax=ax_v_3d, fraction=0.03, pad=0.1, shrink=0.5)

    fig.text(0.02, 0.88, "u conc (2D)", va="center", rotation="vertical", fontsize=11, weight="bold")
    fig.text(0.02, 0.65, "v conc (2D)", va="center", rotation="vertical", fontsize=11, weight="bold")
    fig.text(0.02, 0.42, "u conc (3D)", va="center", rotation="vertical", fontsize=11, weight="bold")
    fig.text(0.02, 0.18, "v conc (3D)", va="center", rotation="vertical", fontsize=11, weight="bold")

    fig.suptitle("Brusselator Evolution", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDESpec
# ---------------------------------------------------------------------------

br_spec = PDESpec(
    name="Brusselator",
    pde_param_keys=["D_u", "D_v", "k1", "k2"],
    create_ic=create_brusselator_ic,
    solve=solve_br,
    field_type=FieldType.PAIRED,
    default_output_name="brusselator",
    history_key="concentration_history",
    prepare_ic_config=prepare_br_ic_config,
    post_sample_params=post_sample_br_params,
    max_physical_magnitude=1e6,
    broad_divergence_catch=True,
    save_visualization=save_brusselator_evolution,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Brusselator training data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    run_generation(br_spec, args.config, args.workers)
