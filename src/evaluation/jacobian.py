"""
Jacobian analysis utilities for PDE operator networks.

Computes partial derivatives of network outputs w.r.t. inputs to extract
learned coefficients and compare against true PDE parameters.

For Navier-Stokes:
    u_t = -u·u_x - v·u_y + ν(u_xx + u_yy) - p_x

    Expected Jacobian entries:
    - ∂u_t/∂u_xx ≈ ν (viscosity)
    - ∂u_t/∂u_yy ≈ ν (viscosity)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.autograd.functional import jvp


# Input feature indices (10 features, shared by NS and Brusselator via dead-inputs approach)
# [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]
FEATURE_NAMES = ["u", "v", "u_x", "u_y", "u_xx", "u_yy", "v_x", "v_y", "v_xx", "v_yy"]
IDX_U = 0
IDX_V = 1
IDX_U_XX = 4
IDX_U_YY = 5
IDX_V_XX = 8
IDX_V_YY = 9

# Legacy aliases for NS
NS_FEATURE_NAMES = FEATURE_NAMES
NS_IDX_U_XX = IDX_U_XX
NS_IDX_U_YY = IDX_U_YY
NS_IDX_V_XX = IDX_V_XX
NS_IDX_V_YY = IDX_V_YY


@dataclass
class JacobianResultsNS:
    """
    Results from Jacobian analysis using JVP (forward-mode autodiff).

    Computes Laplacian coefficient estimates by treating ∇²u = u_xx + u_yy
    as a single mathematical object, using directional derivatives.
    """
    # Per-point Laplacian coefficient estimates
    nu_u: np.ndarray       # shape: (n_samples,) — ν from u equation
    nu_v: np.ndarray       # shape: (n_samples,) — ν from v equation

    # True parameter for comparison
    nu_true: float

    @property
    def nu_recovered_u(self) -> float:
        """Estimated viscosity from u equation only."""
        return float(np.mean(self.nu_u))

    @property
    def nu_recovered_v(self) -> float:
        """Estimated viscosity from v equation only."""
        return float(np.mean(self.nu_v))

    @property
    def error_pct(self) -> float:
        """Relative error in recovered viscosity (%)."""
        if self.nu_true == 0:
            return float('inf')
        return (abs(((self.nu_recovered_v + self.nu_recovered_u) / 2) - self.nu_true) / self.nu_true) * 100


    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'nu_true': self.nu_true,
            'nu_recovered_u_only': self.nu_recovered_u,
            'nu_recovered_v_only': self.nu_recovered_v,
            'error_pct': self.error_pct,
            'nu_u_mean': float(np.mean(self.nu_u)),
            'nu_u_std': float(np.std(self.nu_u)),
            'nu_v_mean': float(np.mean(self.nu_v)),
            'nu_v_std': float(np.std(self.nu_v)),
        }

    def to_npz_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        """Convert to dictionary for NPZ storage (full distributions)."""
        p = f"{prefix}/" if prefix else ""
        return {
            f'{p}nu_u': self.nu_u,
            f'{p}nu_v': self.nu_v,
            f'{p}nu_true': np.array([self.nu_true]),
        }


@dataclass
class JacobianResultsBR:
    """
    Results from Jacobian analysis for Brusselator using JVP.

    Brusselator PDE:
        u_t = D_u·(u_xx + u_yy) + k₁ - (k₂+1)·u + u²·v
        v_t = D_v·(v_xx + v_yy) + k₂·u - u²·v

    Uses directional derivatives to extract Laplacian coefficients:
        D_u = ∂u_t/∂(u_xx + u_yy) = (∂u_t/∂u_xx + ∂u_t/∂u_yy) / 2
        D_v = ∂v_t/∂(v_xx + v_yy) = (∂v_t/∂v_xx + ∂v_t/∂v_yy) / 2
    """
    # Per-point Laplacian coefficient estimates
    D_u: np.ndarray  # shape: (n_samples,) — D_u estimate per point
    D_v: np.ndarray  # shape: (n_samples,) — D_v estimate per point

    # True parameters for comparison
    D_u_true: float
    D_v_true: float

    @property
    def D_u_recovered(self) -> float:
        """Estimated D_u from mean of per-point estimates."""
        return float(np.mean(self.D_u))

    @property
    def D_v_recovered(self) -> float:
        """Estimated D_v from mean of per-point estimates."""
        return float(np.mean(self.D_v))

    @property
    def D_u_error_pct(self) -> float:
        """Relative error in recovered D_u (%)."""
        if self.D_u_true == 0:
            return float('inf')
        return abs(self.D_u_recovered - self.D_u_true) / self.D_u_true * 100

    @property
    def D_v_error_pct(self) -> float:
        """Relative error in recovered D_v (%)."""
        if self.D_v_true == 0:
            return float('inf')
        return abs(self.D_v_recovered - self.D_v_true) / self.D_v_true * 100

    @property
    def error_pct(self) -> float:
        """Average error across both diffusion coefficients."""
        return (self.D_u_error_pct + self.D_v_error_pct) / 2

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'D_u_true': self.D_u_true,
            'D_v_true': self.D_v_true,
            'D_u_recovered': self.D_u_recovered,
            'D_v_recovered': self.D_v_recovered,
            'D_u_error_pct': self.D_u_error_pct,
            'D_v_error_pct': self.D_v_error_pct,
            'error_pct': self.error_pct,
            'D_u_mean': float(np.mean(self.D_u)),
            'D_u_std': float(np.std(self.D_u)),
            'D_v_mean': float(np.mean(self.D_v)),
            'D_v_std': float(np.std(self.D_v)),
        }

    def to_npz_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        """Convert to dictionary for NPZ storage (full distributions)."""
        p = f"{prefix}/" if prefix else ""
        return {
            f'{p}D_u': self.D_u,  # Per-point estimates for histogram
            f'{p}D_v': self.D_v,
            f'{p}D_u_true': np.array([self.D_u_true]),
            f'{p}D_v_true': np.array([self.D_v_true]),
        }


# Union type for either result type
JacobianResultsType = JacobianResultsNS | JacobianResultsBR


def compute_laplacian_jacobian_jvp_ns(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: str = "cpu"
) -> dict:
    """
    Compute Laplacian coefficients using JVP (forward-mode autodiff).

    Uses directional derivatives to treat the Laplacian as a single
    mathematical object. For each sample, computes:
    - nu_u = (∂u_t/∂u_xx + ∂u_t/∂u_yy) / 2  (from u equation)
    - nu_v = (∂v_t/∂v_xx + ∂v_t/∂v_yy) / 2  (from v equation)

    Args:
        model: The neural network
        X: Input tensor of shape (n_samples, 10)
        device: Device to use for computation

    Returns:
        dict with 'nu_u', 'nu_v' arrays (n_samples,)
    """
    model.eval()
    model = model.to(device)
    X = X.to(device)

    # Tangent vectors for Laplacian directions
    # u-Laplacian: perturb u_xx (idx 4) and u_yy (idx 5) together
    tangent_u = torch.zeros_like(X)
    tangent_u[:, IDX_U_XX] = 1.0
    tangent_u[:, IDX_U_YY] = 1.0

    # v-Laplacian: perturb v_xx (idx 8) and v_yy (idx 9) together
    tangent_v = torch.zeros_like(X)
    tangent_v[:, IDX_V_XX] = 1.0
    tangent_v[:, IDX_V_YY] = 1.0

    def forward(x):
        return model(x)

    # Compute JVP for u-Laplacian direction
    # jvp_u[:, 0] = ∂u_t/∂u_xx + ∂u_t/∂u_yy (sensitivity of u_t to u-Laplacian)
    # jvp_u[:, 1] = ∂v_t/∂u_xx + ∂v_t/∂u_yy (cross-term, should be ~0)
    _, jvp_u = jvp(forward, (X,), (tangent_u,))

    # Compute JVP for v-Laplacian direction
    # jvp_v[:, 0] = ∂u_t/∂v_xx + ∂u_t/∂v_yy (cross-term, should be ~0)
    # jvp_v[:, 1] = ∂v_t/∂v_xx + ∂v_t/∂v_yy (sensitivity of v_t to v-Laplacian)
    _, jvp_v = jvp(forward, (X,), (tangent_v,))

    # Extract and divide by 2 to get coefficient estimates
    # (dividing by 2 because we perturbed both xx and yy by 1, so Laplacian changed by 2)
    nu_u = (jvp_u[:, 0] / 2).detach().cpu().numpy()
    nu_v = (jvp_v[:, 1] / 2).detach().cpu().numpy()

    return {
        'nu_u': nu_u,           # ν from u equation only
        'nu_v': nu_v,           # ν from v equation only
    }


def compute_laplacian_jacobian_jvp_br(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: str = "cpu"
) -> dict:
    """
    Compute diffusion coefficients for Brusselator using JVP.

    For Brusselator, D_u and D_v are different, so we compute them separately:
    - D_u = (∂u_t/∂u_xx + ∂u_t/∂u_yy) / 2
    - D_v = (∂v_t/∂v_xx + ∂v_t/∂v_yy) / 2

    Args:
        model: The neural network
        X: Input tensor of shape (n_samples, 10)
        device: Device to use for computation

    Returns:
        dict with 'D_u', 'D_v' arrays (n_samples,)
    """
    model.eval()
    model = model.to(device)
    X = X.to(device)

    # Same tangent vectors as NS
    tangent_u = torch.zeros_like(X)
    tangent_u[:, IDX_U_XX] = 1.0
    tangent_u[:, IDX_U_YY] = 1.0

    tangent_v = torch.zeros_like(X)
    tangent_v[:, IDX_V_XX] = 1.0
    tangent_v[:, IDX_V_YY] = 1.0

    def forward(x):
        return model(x)

    _, jvp_u = jvp(forward, (X,), (tangent_u,))
    _, jvp_v = jvp(forward, (X,), (tangent_v,))

    # For Brusselator, D_u comes from u equation, D_v from v equation
    D_u = (jvp_u[:, 0] / 2).detach().cpu().numpy()
    D_v = (jvp_v[:, 1] / 2).detach().cpu().numpy()

    return {
        'D_u': D_u,
        'D_v': D_v,
    }


def analyze_jacobian_ns(
    model: torch.nn.Module,
    features: np.ndarray,
    nu_true: float,
    device: str = "cpu",
    max_samples: Optional[int] = None
) -> JacobianResultsNS:
    """
    Analyze Jacobian for Navier-Stokes to extract viscosity coefficient.

    Uses JVP (forward-mode autodiff) to compute directional derivatives
    that treat the Laplacian ∇²u = u_xx + u_yy as a single object.

    Args:
        model: Trained PDE operator network
        features: Input features of shape (n_samples, 10)
        nu_true: True viscosity value
        device: Device for computation
        max_samples: Maximum samples to use (None = use all)

    Returns:
        JacobianResults with per-point Laplacian coefficient estimates
    """
    # Subsample if needed
    if max_samples is not None and features.shape[0] > max_samples:
        idx = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[idx]

    X = torch.tensor(features, dtype=torch.float32)

    # Compute Laplacian coefficients using JVP
    results = compute_laplacian_jacobian_jvp_ns(model, X, device=device)

    return JacobianResultsNS(
        nu_u=results['nu_u'],
        nu_v=results['nu_v'],
        nu_true=nu_true,
    )


def analyze_jacobian_br(
    model: torch.nn.Module,
    features: np.ndarray,
    D_u_true: float,
    D_v_true: float,
    device: str = "cpu",
    max_samples: Optional[int] = None
) -> JacobianResultsBR:
    """
    Analyze Jacobian for Brusselator to extract diffusion coefficients.

    Uses JVP (forward-mode autodiff) to compute directional derivatives
    that treat the Laplacians as single objects.

    Args:
        model: Trained PDE operator network
        features: Input features of shape (n_samples, 10)
        D_u_true: True diffusion coefficient for u
        D_v_true: True diffusion coefficient for v
        device: Device for computation
        max_samples: Maximum samples to use (None = use all)

    Returns:
        BrusselatorJacobianResults with per-point coefficient estimates
    """
    # Subsample if needed
    if max_samples is not None and features.shape[0] > max_samples:
        idx = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[idx]

    X = torch.tensor(features, dtype=torch.float32)

    # Compute diffusion coefficients using JVP
    results = compute_laplacian_jacobian_jvp_br(model, X, device=device)

    return JacobianResultsBR(
        D_u=results['D_u'],
        D_v=results['D_v'],
        D_u_true=D_u_true,
        D_v_true=D_v_true,
    )
