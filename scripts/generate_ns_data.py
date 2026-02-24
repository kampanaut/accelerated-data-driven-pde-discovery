"""
Generate Navier-Stokes training data for PDE discovery.

Solves incompressible NS equations for each IC, FFTs velocity snapshots to
Fourier coefficients, FFTs pressure snapshots to p_hat, saves .npz files.

Usage:
    python scripts/generate_ns_data.py --config configs/ns_train-1.yaml
    python scripts/generate_ns_data.py --config configs/ns_train-1.yaml --workers 4
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.data.generation import PDESpec, FieldType, run_generation
from src.data.initial_conditions_ns import create_ns_ic
from src.pde.navier_stokes import solve_ns


# ---------------------------------------------------------------------------
# Post-Fourier hook: FFT pressure snapshots → p_hat
# ---------------------------------------------------------------------------

def ns_post_fourier(fourier_data: dict, solver_results: dict) -> dict:
    """Add p_hat to fourier_data from solver's pressure_history."""
    pressure_history = solver_results["pressure_history"]
    n = len(pressure_history)
    ny, nx = pressure_history[0].shape
    p_hat = np.empty((n, ny, nx), dtype=np.complex128)
    for i, p in enumerate(pressure_history):
        p_hat[i] = np.fft.fft2(p)
    print(f"  FFT'd {n} pressure snapshots → p_hat")
    fourier_data["p_hat"] = p_hat
    return fourier_data


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_flow_evolution(
    velocity_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    output_path: str,
    n_snapshots: int = 8,
) -> None:
    """
    Save a multi-panel figure showing flow evolution over time.

    Creates a 5xn_snapshots grid:
    - Row 1: Velocity vector fields (2D)
    - Row 2: Vorticity contours (2D)
    - Row 3: Vorticity surfaces (3D)
    - Row 4: Velocity magnitude contours (2D)
    - Row 5: Velocity magnitude surfaces (3D)
    """
    indices = np.linspace(0, len(velocity_history) - 1, n_snapshots, dtype=int)

    fig = plt.figure(figsize=(5 * n_snapshots, 25))

    for col, idx in enumerate(indices):
        u, v = velocity_history[idx]
        t = times[idx]

        if x.ndim == 1 and y.ndim == 1:
            x_grid, y_grid = np.meshgrid(x, y)
        else:
            x_grid, y_grid = x, y

        from src.data.derivatives import compute_vorticity

        vorticity = compute_vorticity(u, v, dx, dy)
        vel_mag = np.sqrt(u**2 + v**2)

        # Row 1: Velocity field (2D quiver)
        ax_vel = plt.subplot(5, n_snapshots, col + 1)
        q_step = max(4, x_grid.shape[0] // 32)
        ax_vel.quiver(
            x_grid[::q_step, ::q_step],
            y_grid[::q_step, ::q_step],
            u[::q_step, ::q_step],
            v[::q_step, ::q_step],
            vel_mag[::q_step, ::q_step],
            cmap="viridis",
        )
        ax_vel.set_title(f"t={t:.3f}")
        ax_vel.set_aspect("equal")

        # Row 2: Vorticity (2D contour)
        ax_vort_2d = plt.subplot(5, n_snapshots, n_snapshots + col + 1)
        contour_vort = ax_vort_2d.contourf(
            x_grid, y_grid, vorticity, levels=20, cmap="RdBu_r"
        )
        plt.colorbar(contour_vort, ax=ax_vort_2d, fraction=0.046, pad=0.04)
        ax_vort_2d.set_aspect("equal")

        # Row 3: Vorticity (3D surface)
        ax_vort_3d = plt.subplot(
            5, n_snapshots, 2 * n_snapshots + col + 1, projection="3d"
        )
        assert isinstance(ax_vort_3d, Axes3D)
        step = max(1, x_grid.shape[0] // 32)
        surf_vort = ax_vort_3d.plot_surface(
            x_grid[::step, ::step],
            y_grid[::step, ::step],
            vorticity[::step, ::step],
            cmap="RdBu_r",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        ax_vort_3d.set_xlabel("x")
        ax_vort_3d.set_ylabel("y")
        ax_vort_3d.set_zlabel("\u03c9")
        ax_vort_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_vort, ax=ax_vort_3d, fraction=0.03, pad=0.1, shrink=0.5)

        # Row 4: Velocity magnitude (2D contour)
        ax_mag_2d = plt.subplot(5, n_snapshots, 3 * n_snapshots + col + 1)
        contour_mag = ax_mag_2d.contourf(
            x_grid, y_grid, vel_mag, levels=20, cmap="viridis"
        )
        plt.colorbar(contour_mag, ax=ax_mag_2d, fraction=0.046, pad=0.04)
        ax_mag_2d.set_aspect("equal")

        # Row 5: Velocity magnitude (3D surface)
        ax_mag_3d = plt.subplot(
            5, n_snapshots, 4 * n_snapshots + col + 1, projection="3d"
        )
        assert isinstance(ax_mag_3d, Axes3D)
        surf_mag = ax_mag_3d.plot_surface(
            x_grid[::step, ::step],
            y_grid[::step, ::step],
            vel_mag[::step, ::step],
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        ax_mag_3d.set_xlabel("x")
        ax_mag_3d.set_ylabel("y")
        ax_mag_3d.set_zlabel("|v|")
        ax_mag_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_mag, ax=ax_mag_3d, fraction=0.03, pad=0.1, shrink=0.5)

    # Row labels
    fig.text(0.02, 0.88, "Velocity vectors", va="center", rotation="vertical", fontsize=11, weight="bold")
    fig.text(0.02, 0.72, "Vorticity (2D)", va="center", rotation="vertical", fontsize=11, weight="bold")
    fig.text(0.02, 0.55, "Vorticity (3D)", va="center", rotation="vertical", fontsize=11, weight="bold")
    fig.text(0.02, 0.38, "Velocity mag (2D)", va="center", rotation="vertical", fontsize=11, weight="bold")
    fig.text(0.02, 0.20, "Velocity mag (3D)", va="center", rotation="vertical", fontsize=11, weight="bold")

    fig.suptitle("Flow Evolution", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDESpec
# ---------------------------------------------------------------------------

ns_spec = PDESpec(
    name="Navier-Stokes",
    pde_param_keys=["nu"],
    create_ic=create_ns_ic,
    solve=solve_ns,
    field_type=FieldType.PAIRED,
    default_output_name="navier_stokes",
    history_key="velocity_history",
    post_fourier=ns_post_fourier,
    extra_save_keys=[("nu_used", "nu")],
    save_visualization=save_flow_evolution,
    vis_needs_grid_spacing=True,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Navier-Stokes training data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    run_generation(ns_spec, args.config, args.workers)
