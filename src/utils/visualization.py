"""
Visualization utilities for fluid flow fields.

Provides functions to visualize velocity fields, vorticity, and derivatives.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as pltf
from typing import Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D


def plot_velocity_field(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    time: float,
    title: Optional[str] = None,
    skip: int = 4,
    figsize: Tuple[float, float] = (10, 8),
) -> pltf.Figure:
    """
    Plot velocity field as quiver plot.

    Args:
        u: x-component of velocity, shape (ny, nx)
        v: y-component of velocity, shape (ny, nx)
        x: x-coordinates, shape (ny, nx) or (nx,)
        y: y-coordinates, shape (ny, nx) or (ny,)
        time: Time value for the snapshot
        title: Optional custom title
        skip: Plot every `skip`-th arrow (for clarity)
        figsize: Figure size in inches

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle 1D coordinate arrays
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y)
    else:
        x_grid, y_grid = x, y

    # Plot velocity field (subsample for clarity)
    quiver = ax.quiver(
        x_grid[::skip, ::skip],
        y_grid[::skip, ::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
        np.sqrt(u[::skip, ::skip] ** 2 + v[::skip, ::skip] ** 2),  # Color by magnitude
        cmap="viridis",
    )

    plt.colorbar(quiver, ax=ax, label="Velocity magnitude")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Velocity field at t={time:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    return fig


def plot_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    time: float,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> pltf.Figure:
    """
    Plot vorticity field as contour plot.

    Args:
        u: x-component of velocity, shape (ny, nx)
        v: y-component of velocity, shape (ny, nx)
        x: x-coordinates, shape (ny, nx) or (nx,)
        y: y-coordinates, shape (ny, nx) or (ny,)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        time: Time value for the snapshot
        title: Optional custom title
        figsize: Figure size in inches

    Returns:
        Matplotlib figure object
    """
    from src.data.derivatives import compute_vorticity

    fig, ax = plt.subplots(figsize=figsize)

    # Compute vorticity
    vorticity = compute_vorticity(u, v, dx, dy)

    # Handle 1D coordinate arrays
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y)
    else:
        x_grid, y_grid = x, y

    # Plot vorticity contours
    levels = 20
    contour = ax.contourf(x_grid, y_grid, vorticity, levels=levels, cmap="RdBu_r")
    plt.colorbar(contour, ax=ax, label="Vorticity (ω)")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Vorticity at t={time:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    return fig


def plot_field_comparison(
    fields: list,
    labels: list,
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Field Comparison",
    figsize: Optional[Tuple[float, float]] = None,
) -> pltf.Figure:
    """
    Plot multiple scalar fields side by side for comparison.

    Args:
        fields: List of 2D arrays to plot
        labels: List of labels for each field
        x: x-coordinates
        y: y-coordinates
        title: Overall figure title
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure object
    """
    n_fields = len(fields)

    if figsize is None:
        figsize = (6 * n_fields, 5)

    fig, axes = plt.subplots(1, n_fields, figsize=figsize)

    if n_fields == 1:
        axes = [axes]

    # Handle 1D coordinate arrays
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y)
    else:
        x_grid, y_grid = x, y

    for ax, field, label in zip(axes, fields, labels):
        im = ax.contourf(x_grid, y_grid, field, levels=20, cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def save_flow_evolution(
    velocity_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    output_path: str,
    n_snapshots: int = 4,
):
    """
    Save a multi-panel figure showing flow evolution over time.

    Creates a 5×n_snapshots grid:
    - Row 1: Velocity vector fields (2D)
    - Row 2: Vorticity contours (2D)
    - Row 3: Vorticity surfaces (3D)
    - Row 4: Velocity magnitude contours (2D)
    - Row 5: Velocity magnitude surfaces (3D)

    Args:
        velocity_history: List of (u, v) tuples at different times
        times: Array of time values
        x, y: Coordinate arrays
        dx, dy: Grid spacings
        output_path: Path to save the figure
        n_snapshots: Number of snapshots to show
    """

    indices = np.linspace(0, len(velocity_history) - 1, n_snapshots, dtype=int)

    fig = plt.figure(figsize=(5 * n_snapshots, 25))

    for col, idx in enumerate(indices):
        u, v = velocity_history[idx]
        t = times[idx]

        # Prepare grid
        if x.ndim == 1 and y.ndim == 1:
            x_grid, y_grid = np.meshgrid(x, y)
        else:
            x_grid, y_grid = x, y

        # Compute derived fields
        from src.data.derivatives import compute_vorticity

        vorticity = compute_vorticity(u, v, dx, dy)
        vel_mag = np.sqrt(u**2 + v**2)

        # Row 1: Velocity field (2D quiver)
        ax_vel = plt.subplot(5, n_snapshots, col + 1)
        _ = ax_vel.quiver( # quiver
            x_grid[::4, ::4],
            y_grid[::4, ::4],
            u[::4, ::4],
            v[::4, ::4],
            vel_mag[::4, ::4],
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
        surf_vort = ax_vort_3d.plot_surface(
            x_grid,
            y_grid,
            vorticity,
            cmap="RdBu_r",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )
        ax_vort_3d.set_xlabel("x")
        ax_vort_3d.set_ylabel("y")
        ax_vort_3d.set_zlabel("ω")
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
            x_grid,
            y_grid,
            vel_mag,
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

    # Add row labels
    fig.text(
        0.02,
        0.88,
        "Velocity vectors",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.02,
        0.72,
        "Vorticity (2D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.02,
        0.55,
        "Vorticity (3D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.02,
        0.38,
        "Velocity mag (2D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.02,
        0.20,
        "Velocity mag (3D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )

    fig.suptitle("Flow Evolution", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved flow evolution visualization to {output_path}")
