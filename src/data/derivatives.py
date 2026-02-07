"""
Finite difference computation for spatial and temporal derivatives.

This module provides functions to compute derivatives using central finite differences.
The derivatives are used as inputs to the Neural Network N for PDE discovery.
"""

import numpy as np
from typing import Tuple


def spatial_derivatives(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute first and second spatial derivatives using central finite differences.

    Uses numpy.gradient which implements 2nd-order accurate central differences:
    - Interior points: f'(x) = (f(x+h) - f(x-h)) / (2h)
    - Boundary points: forward/backward differences

    Args:
        field: 2D array of shape (ny, nx) representing a scalar field
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Tuple of (field_x, field_y, field_xx, field_yy) where:
        - field_x: ∂field/∂x
        - field_y: ∂field/∂y
        - field_xx: ∂²field/∂x²
        - field_yy: ∂²field/∂y²
    """
    # First derivatives
    field_x = np.gradient(field, dx, axis=1)  # x is axis=1 (columns)
    field_y = np.gradient(field, dy, axis=0)  # y is axis=0 (rows)

    # Second derivatives (apply gradient again)
    field_xx = np.gradient(field_x, dx, axis=1)
    field_yy = np.gradient(field_y, dy, axis=0)

    return field_x, field_y, field_xx, field_yy


def spectral_spatial_derivatives(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute first and second spatial derivatives using FFT (spectral method).

    Assumes periodic boundary conditions. Exact to machine precision for
    band-limited signals (i.e., signals representable on the grid).

    Derivative in Fourier space: multiply coefficients by (i*k)^n
    - First derivative: i*k * f_hat
    - Second derivative: -k^2 * f_hat

    Args:
        field: 2D array of shape (ny, nx) representing a scalar field
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        Tuple of (field_x, field_y, field_xx, field_yy)
    """
    ny, nx = field.shape

    # Wavenumbers (angular)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)

    # 2D FFT
    f_hat = np.fft.fft2(field)

    # First derivatives
    field_x = np.real(np.fft.ifft2(1j * kx[np.newaxis, :] * f_hat))
    field_y = np.real(np.fft.ifft2(1j * ky[:, np.newaxis] * f_hat))

    # Second derivatives
    field_xx = np.real(np.fft.ifft2(-kx[np.newaxis, :] ** 2 * f_hat))
    field_yy = np.real(np.fft.ifft2(-ky[:, np.newaxis] ** 2 * f_hat))

    return field_x, field_y, field_xx, field_yy


def temporal_derivative(field_history: list, dt: float) -> np.ndarray:
    """
    Compute time derivative from a sequence of snapshots using central differences.

    Uses central difference formula:
    f_t(t) = (f(t+dt) - f(t-dt)) / (2*dt)

    Note: This requires snapshots at t-dt, t, and t+dt, so we lose the first
    and last timesteps.

    Args:
        field_history: List of 2D arrays representing field at different times
                       Ordered as [field(t=0), field(t=dt), field(t=2dt), ...]
        dt: Time spacing between snapshots

    Returns:
        Array of shape (n_timesteps-2, ny, nx) containing time derivatives.
        The returned array contains derivatives for timesteps 1 through n-2.
    """
    n_timesteps = len(field_history)

    if n_timesteps < 3:
        raise ValueError(f"Need at least 3 timesteps for central differences, got {n_timesteps}")

    # Preallocate array for time derivatives (excluding first and last timesteps)
    field_t = np.zeros((n_timesteps - 2, *field_history[0].shape))

    # Central difference for each interior timestep
    for i in range(1, n_timesteps - 1):
        field_t[i - 1] = (field_history[i + 1] - field_history[i - 1]) / (2 * dt)

    return field_t


def compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute vorticity from velocity components.

    Vorticity is defined as:
    ω = ∂v/∂x - ∂u/∂y

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
