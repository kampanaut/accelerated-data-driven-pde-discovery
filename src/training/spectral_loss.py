"""
Spectral structural loss via NUFFT Type 1.

Extracts Fourier coefficients from MLP predictions at scattered collocation
points (no grid required), then compares per-mode against the true target's
coefficients. Combined with pointwise MSE, this penalizes spectral errors
that pointwise loss alone might miss.

Uses pytorch-finufft for autograd-compatible NUFFT on GPU.
"""

import torch
import torch.nn.functional as F
from pytorch_finufft.functional import finufft_type1


def compute_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_pts: torch.Tensor,
    y_pts: torch.Tensor,
    Lx: float,
    Ly: float,
    n_modes: int,
) -> torch.Tensor:
    """
    Compute spectral loss between predictions and targets at scattered points.

    Uses NUFFT Type 1 to extract Fourier coefficients from both pred and target
    at the same scattered locations, then compares per-mode.

    Args:
        pred: Model predictions at collocation points, shape (N, n_channels)
        target: True targets at collocation points, shape (N, n_channels)
        x_pts: x-coordinates of collocation points, shape (N,)
        y_pts: y-coordinates of collocation points, shape (N,)
        Lx: Domain length in x
        Ly: Domain length in y
        n_modes: Number of Fourier modes per dimension (M×M grid)

    Returns:
        Scalar spectral loss tensor (differentiable)
    """
    # Scale coordinates to NUFFT convention: [-π, π)
    x_scaled = ((2.0 * torch.pi * (x_pts / Lx)) - torch.pi).to(pred.dtype)
    y_scaled = ((2.0 * torch.pi * (y_pts / Ly)) - torch.pi).to(pred.dtype)

    points = torch.stack([x_scaled, y_scaled], dim=0)  # (D, N) where D=2
    output_shape = (n_modes, n_modes)

    # NUFFT requires complex values; batch dim = all dims except last (which must be N)
    # pred is (N, C) → transpose to (C, N) so C is batch dim
    pred_c = torch.complex(pred.T, torch.zeros_like(pred.T)).contiguous()
    target_c = torch.complex(target.T, torch.zeros_like(target.T)).contiguous()

    pred_hat = finufft_type1(points, pred_c, output_shape)  # (C, M, M)
    true_hat = finufft_type1(points, target_c, output_shape)  # (C, M, M)

    # view_as_real: (C, M, M) complex → (C, M, M, 2) real — F.mse_loss works on real
    pred_r = torch.view_as_real(pred_hat)
    true_r = torch.view_as_real(true_hat)

    loss = F.mse_loss(pred_r, true_r) / (true_r**2).mean()

    return loss
