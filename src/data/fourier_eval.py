"""
Fourier collocation evaluation for PDE data (Brusselator, Lambda-Omega, Navier-Stokes, FitzHugh-Nagumo).

Evaluates 2D periodic fields (stored as FFT coefficients) at arbitrary
spatial points using DFT synthesis. Provides spectrally exact derivatives
via wavenumber multiplication.

The key formula (unnormalized DFT convention matching numpy.fft):
    f(x, y) = (1 / (nx * ny)) * sum_{j,k} f_hat[j,k] * exp(i * kx_j * x) * exp(i * ky_k * y)

where kx_j = 2*pi*fftfreq(nx, dx) are angular wavenumbers.
"""

import torch
from typing import Tuple


def build_wavenumbers(
    nx: int, ny: int, Lx: float, Ly: float, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build angular wavenumber arrays for a periodic domain.

    Args:
        nx, ny: Grid dimensions
        Lx, Ly: Domain lengths
        device: Torch device

    Returns:
        (kx, ky): Angular wavenumber tensors, shapes (nx,) and (ny,), complex64
    """
    dx = Lx / nx
    dy = Ly / ny
    kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=dx, device=device, dtype=torch.float64)
    ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=dy, device=device, dtype=torch.float64)
    return (kx, ky)


def fourier_eval_2d(
    f_hat: torch.Tensor,
    E_x: torch.Tensor,
    E_y: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate a 2D Fourier series at arbitrary points using precomputed phase matrices.

    Args:
        f_hat: 2D FFT coefficients, shape (ny, nx), complex128
        E_x: x phase matrix, shape (n_points, nx), complex128
        E_y: y phase matrix, shape (n_points, ny), complex128

    Returns:
        Real values at the query points, shape (n_points,), float64
    """
    ny, nx = f_hat.shape
    tmp = f_hat @ E_x.T                          # (ny, n_points)
    result = torch.sum(E_y.T * tmp, dim=0) / (nx * ny)
    return result.real


def evaluate_br_features(
    u_hat: torch.Tensor,
    v_hat: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    E_x: torch.Tensor,
    E_y: torch.Tensor,
    D_u: float,
    D_v: float,
    k1: float,
    k2: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate all 10 features and 2 PDE-RHS targets at arbitrary points.

    Features: [u, v, u_x, A_y, u_xx, u_yy, B_x, B_y, v_xx, v_yy]
    Targets: [u_t, v_t] from PDE RHS

    All computation on GPU. Returns float32 tensors.
    """
    _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y)

    ikx = 1j * kx.unsqueeze(0)         # (1, nx)
    iky = 1j * ky.unsqueeze(1)         # (ny, 1)
    neg_kx2 = -(kx.unsqueeze(0)) ** 2  # (1, nx)
    neg_ky2 = -(ky.unsqueeze(1)) ** 2  # (ny, 1)

    u = _eval(u_hat)
    v = _eval(v_hat)
    u_x = _eval(ikx * u_hat)
    u_y = _eval(iky * u_hat)
    v_x = _eval(ikx * v_hat)
    v_y = _eval(iky * v_hat)
    u_xx = _eval(neg_kx2 * u_hat)
    u_yy = _eval(neg_ky2 * u_hat)
    v_xx = _eval(neg_kx2 * v_hat)
    v_yy = _eval(neg_ky2 * v_hat)

    features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

    u_sq_v = u ** 2 * v
    u_t = D_u * (u_xx + u_yy) + k1 - (k2 + 1) * u + u_sq_v
    v_t = D_v * (v_xx + v_yy) + k2 * u - u_sq_v
    targets = torch.stack([u_t, v_t], dim=1)

    return features.float(), targets.float()


def evaluate_lo_features(
    u_hat: torch.Tensor,
    v_hat: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    E_x: torch.Tensor,
    E_y: torch.Tensor,
    D_u: float,
    D_v: float,
    a: float,
    c: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate all 10 features and 2 PDE-RHS targets for Lambda-Omega.

    Features: [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]
    Targets: [u_t, v_t] from PDE RHS:
        r2 = u^2 + v^2
        u_t = D_u*(u_xx + u_yy) + a*u - (u + c*v)*r2
        v_t = D_v*(v_xx + v_yy) + a*v + (c*u - v)*r2

    Args:
        u_hat, v_hat: Fourier coefficients for one snapshot, shape (ny, nx), complex128
        kx, ky: Wavenumber arrays, shapes (nx,) and (ny,)
        E_x, E_y: Phase matrices, shapes (n_points, nx) and (n_points, ny)
        D_u, D_v: Diffusion coefficients
        a: Linear growth rate
        c: Rotation parameter

    Returns:
        (features, targets): float32 tensors, shapes (n_points, 10) and (n_points, 2)
    """
    _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y)

    ikx = 1j * kx.unsqueeze(0)         # (1, nx)
    iky = 1j * ky.unsqueeze(1)         # (ny, 1)
    neg_kx2 = -(kx.unsqueeze(0)) ** 2  # (1, nx)
    neg_ky2 = -(ky.unsqueeze(1)) ** 2  # (ny, 1)

    u = _eval(u_hat)
    v = _eval(v_hat)
    u_x = _eval(ikx * u_hat)
    u_y = _eval(iky * u_hat)
    v_x = _eval(ikx * v_hat)
    v_y = _eval(iky * v_hat)
    u_xx = _eval(neg_kx2 * u_hat)
    u_yy = _eval(neg_ky2 * u_hat)
    v_xx = _eval(neg_kx2 * v_hat)
    v_yy = _eval(neg_ky2 * v_hat)

    features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

    r2 = u ** 2 + v ** 2
    u_t = D_u * (u_xx + u_yy) + a * u - (u + c * v) * r2
    v_t = D_v * (v_xx + v_yy) + a * v + (c * u - v) * r2
    targets = torch.stack([u_t, v_t], dim=1)

    return features.float(), targets.float()


def evaluate_ns_features(
    u_hat: torch.Tensor,
    v_hat: torch.Tensor,
    p_hat: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    E_x: torch.Tensor,
    E_y: torch.Tensor,
    nu: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate all 10 features and 2 PDE-RHS targets for Navier-Stokes
    at arbitrary points.

    Features: [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]
    Targets: [u_t, v_t] from NS momentum equation:
        u_t = -(u*u_x + v*u_y) - p_x + nu*(u_xx + u_yy)
        v_t = -(u*v_x + v*v_y) - p_y + nu*(v_xx + v_yy)

    Args:
        u_hat, v_hat: Fourier coefficients for one snapshot, shape (ny, nx), complex128
        p_hat: Pressure Fourier coefficients, shape (ny, nx), complex128
        kx, ky: Wavenumber arrays, shapes (nx,) and (ny,)
        E_x, E_y: Phase matrices, shapes (n_points, nx) and (n_points, ny)
        nu: Kinematic viscosity

    Returns:
        (features, targets): float32 tensors, shapes (n_points, 10) and (n_points, 2)
    """
    _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y)

    ikx = 1j * kx.unsqueeze(0)         # (1, nx)
    iky = 1j * ky.unsqueeze(1)         # (ny, 1)
    neg_kx2 = -(kx.unsqueeze(0)) ** 2  # (1, nx)
    neg_ky2 = -(ky.unsqueeze(1)) ** 2  # (ny, 1)

    u = _eval(u_hat)
    v = _eval(v_hat)
    u_x = _eval(ikx * u_hat)
    u_y = _eval(iky * u_hat)
    v_x = _eval(ikx * v_hat)
    v_y = _eval(iky * v_hat)
    u_xx = _eval(neg_kx2 * u_hat)
    u_yy = _eval(neg_ky2 * u_hat)
    v_xx = _eval(neg_kx2 * v_hat)
    v_yy = _eval(neg_ky2 * v_hat)

    # Pressure gradient from p_hat
    p_x = _eval(ikx * p_hat)
    p_y = _eval(iky * p_hat)

    features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

    # NS momentum equation
    u_t = -(u * u_x + v * u_y) - p_x + nu * (u_xx + u_yy)
    v_t = -(u * v_x + v * v_y) - p_y + nu * (v_xx + v_yy)
    targets = torch.stack([u_t, v_t], dim=1)

    return features.float(), targets.float()


def evaluate_fhn_features(
    u_hat: torch.Tensor,
    v_hat: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    E_x: torch.Tensor,
    E_y: torch.Tensor,
    D_u: float,
    D_v: float,
    eps: float,
    a: float,
    b: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate all 10 features and 2 PDE-RHS targets for FitzHugh-Nagumo.

    Features: [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]
    Targets: [u_t, v_t] from PDE RHS:
        u_t = D_u*(u_xx + u_yy) + u - u^3 - v
        v_t = D_v*(v_xx + v_yy) + eps*(u - a*v - b)

    Args:
        u_hat, v_hat: Fourier coefficients for one snapshot, shape (ny, nx), complex128
        kx, ky: Wavenumber arrays, shapes (nx,) and (ny,)
        E_x, E_y: Phase matrices, shapes (n_points, nx) and (n_points, ny)
        D_u, D_v: Diffusion coefficients
        eps: Timescale separation
        a: Recovery coupling
        b: Excitability threshold

    Returns:
        (features, targets): float32 tensors, shapes (n_points, 10) and (n_points, 2)
    """
    _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y)

    ikx = 1j * kx.unsqueeze(0)         # (1, nx)
    iky = 1j * ky.unsqueeze(1)         # (ny, 1)
    neg_kx2 = -(kx.unsqueeze(0)) ** 2  # (1, nx)
    neg_ky2 = -(ky.unsqueeze(1)) ** 2  # (ny, 1)

    u = _eval(u_hat)
    v = _eval(v_hat)
    u_x = _eval(ikx * u_hat)
    u_y = _eval(iky * u_hat)
    v_x = _eval(ikx * v_hat)
    v_y = _eval(iky * v_hat)
    u_xx = _eval(neg_kx2 * u_hat)
    u_yy = _eval(neg_ky2 * u_hat)
    v_xx = _eval(neg_kx2 * v_hat)
    v_yy = _eval(neg_ky2 * v_hat)

    features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

    u_t = D_u * (u_xx + u_yy) + u - u ** 3 - v
    v_t = D_v * (v_xx + v_yy) + eps * (u - a * v - b)
    targets = torch.stack([u_t, v_t], dim=1)

    return features.float(), targets.float()


def evaluate_heat_features(
    u_hat: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    E_x: torch.Tensor,
    E_y: torch.Tensor,
    D: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate 5 features and 1 PDE-RHS target for the heat equation u_t = D*(u_xx + u_yy).

    Features: [u, u_x, u_y, u_xx, u_yy]
    Targets: [u_t] from PDE RHS

    Args:
        u_hat: Fourier coefficients for one snapshot, shape (ny, nx), complex128
        kx, ky: Wavenumber arrays, shapes (nx,) and (ny,)
        E_x, E_y: Phase matrices, shapes (n_points, nx) and (n_points, ny)
        D: Diffusion coefficient

    Returns:
        (features, targets): float32 tensors, shapes (n_points, 5) and (n_points, 1)
    """

    _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y)

    ikx = 1j * kx.unsqueeze(0)         # (1, nx)
    iky = 1j * ky.unsqueeze(1)         # (ny, 1)
    neg_kx2 = -(kx.unsqueeze(0)) ** 2  # (1, nx)
    neg_ky2 = -(ky.unsqueeze(1)) ** 2  # (ny, 1)

    u = _eval(u_hat)
    u_x = _eval(ikx * u_hat)
    u_y = _eval(iky * u_hat)
    u_xx = _eval(neg_kx2 * u_hat)
    u_yy = _eval(neg_ky2 * u_hat)

    features = torch.stack([u, u_x, u_y, u_xx, u_yy], dim=1)
    u_t = D * (u_xx + u_yy)
    targets = torch.stack([u_t], dim=1)

    return features.float(), targets.float()


def evaluate_nl_heat_features(
    u_hat: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    E_x: torch.Tensor,
    E_y: torch.Tensor,
    K: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate 5 features and 1 PDE-RHS target for the nonlinear heat equation u_t = K*(1-u)*(u_xx + u_yy).

    Features: [u, u_x, u_y, u_xx, u_yy]
    Targets: [u_t] from PDE RHS

    Args:
        u_hat: Fourier coefficients for one snapshot, shape (ny, nx), complex128
        kx, ky: Wavenumber arrays, shapes (nx,) and (ny,)
        E_x, E_y: Phase matrices, shapes (n_points, nx) and (n_points, ny)
        K: Nonlinear diffusion coefficient

    Returns:
        (features, targets): float32 tensors, shapes (n_points, 5) and (n_points, 1)
    """
    _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y)

    ikx = 1j * kx.unsqueeze(0)         # (1, nx)
    iky = 1j * ky.unsqueeze(1)         # (ny, 1)
    neg_kx2 = -(kx.unsqueeze(0)) ** 2  # (1, nx)
    neg_ky2 = -(ky.unsqueeze(1)) ** 2  # (ny, 1)

    u = _eval(u_hat)
    u_x = _eval(ikx * u_hat)
    u_y = _eval(iky * u_hat)
    u_xx = _eval(neg_kx2 * u_hat)
    u_yy = _eval(neg_ky2 * u_hat)

    features = torch.stack([u, u_x, u_y, u_xx, u_yy], dim=1)
    u_t = K * (1 - u) * (u_xx + u_yy)
    targets = torch.stack([u_t], dim=1)

    return features.float(), targets.float()
