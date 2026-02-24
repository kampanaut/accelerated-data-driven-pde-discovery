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
    device: str
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
    f_hat = f_hat.to(device=device)
    ny, nx = f_hat.shape
    tmp = f_hat @ E_x.T                          # (ny, n_points)
    result = torch.sum(E_y.T * tmp, dim=0) / (nx * ny)
    return result.real
