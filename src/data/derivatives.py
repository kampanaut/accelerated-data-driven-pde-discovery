"""
Utility functions for derived quantities from PDE fields.

Currently provides vorticity computation for Navier-Stokes visualization.
Spatial/temporal derivative computation is handled by Fourier evaluation
(see fourier_eval.py) — no finite-difference derivatives needed in the pipeline.
"""

import numpy as np


def compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute vorticity from velocity components.

    Vorticity is defined as:
    omega = dv/dx - du/dy

    Args:
        u: x-component of velocity, shape (ny, nx)
        v: y-component of velocity, shape (ny, nx)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Vorticity field, shape (ny, nx)
    """
    v_x = np.gradient(v, dx, axis=1)
    u_y = np.gradient(u, dy, axis=0)
    return v_x - u_y
