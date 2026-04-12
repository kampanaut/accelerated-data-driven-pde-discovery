"""
Jacobian analysis utilities for PDE operator networks.

Extracts learned coefficients from a trained network by computing directional
derivatives (JVP) along specified input directions. The caller defines *what*
to extract via CoefficientSpec entries; this module handles the mechanics.

Example — extracting D_u from a reaction-diffusion PDE:
    The PDE has u_t = D_u·(u_xx + u_yy) + f(u,v).
    Perturbing u_xx and u_yy together by 1 each changes the Laplacian by 2,
    so JVP / 2 gives the coefficient D_u.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import torch
from torch.autograd.functional import jvp

from src.training.task_loader import CoefficientSpec


@dataclass
class RegressionResult:
    """Result from regression-based coefficient extraction (e.g. NLHeat K)."""
    value: float         # recovered coefficient (K)
    r_squared: float     # R² of the regression fit
    raw_jvp: NDArray[np.floating]     # raw JVP per point (before regression)
    regressor: NDArray[np.floating]   # (1-u) per point — x-axis for scatter


@dataclass
class JacobianResults:
    """Generic results from JVP-based coefficient extraction.

    Stores per-point estimates and true values for an arbitrary set of
    coefficients defined by CoefficientSpec entries.
    """

    estimates: dict[str, NDArray[np.floating]] = field(default_factory=dict)
    true_values: dict[str, float] = field(default_factory=dict)
    regressions: dict[str, RegressionResult] = field(default_factory=dict)

    def recovered(self, name: str) -> float:
        """Mean of per-point estimates for a named coefficient."""
        return float(np.mean(self.estimates[name]))

    def coeff_error_pct(self, name: str) -> float:
        """Relative error (%) for a named coefficient."""
        true = self.true_values[name]
        if true == 0:
            return float("inf")
        return (abs(self.recovered(name) - true) / true) * 100

    @property
    def error_pct(self) -> float:
        """Mean relative error (%) across all coefficients."""
        return float(np.mean([self.coeff_error_pct(n) for n in self.true_values]))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d: dict = {}
        for name in self.estimates:
            d[f"{name}_true"] = self.true_values[name]
            d[f"{name}_recovered"] = self.recovered(name)
            d[f"{name}_error_pct"] = self.coeff_error_pct(name)
            d[f"{name}_mean"] = float(np.mean(self.estimates[name]))
            d[f"{name}_std"] = float(np.std(self.estimates[name]))
        d["error_pct"] = self.error_pct
        for name, reg in self.regressions.items():
            d[f"{name}_regression_r2"] = reg.r_squared
        return d

    def to_npz_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Convert to dictionary for NPZ storage (full distributions)."""
        p = f"{prefix}/" if prefix else ""
        d: dict[str, np.ndarray] = {}
        for name, arr in self.estimates.items():
            d[f"{p}{name}"] = arr
            d[f"{p}{name}_true"] = np.array([self.true_values[name]])
        for name, reg in self.regressions.items():
            d[f"{p}{name}_regression_value"] = np.array([reg.value])
            d[f"{p}{name}_regression_r2"] = np.array([reg.r_squared])
            d[f"{p}{name}_regression_raw_jvp"] = reg.raw_jvp
            d[f"{p}{name}_regression_regressor"] = reg.regressor
        return d


def analyze_jacobian(
    model: torch.nn.Module,
    features: torch.Tensor,
    specs: list[CoefficientSpec],
    device: str = "cpu",
    max_samples: Optional[int] = None,
) -> JacobianResults:
    """
    Extract coefficients from a trained PDE operator network via JVP.

    For each CoefficientSpec, constructs a tangent vector that perturbs the
    specified input indices, computes the JVP, reads the specified output
    component, and divides by the number of perturbed indices.

    Args:
        model: Trained PDE operator network
        features: Input features tensor (n_samples, n_features) on device
        specs: List of CoefficientSpec defining what to extract
        device: Device for computation
        max_samples: Maximum samples to use (None = use all)

    Returns:
        JacobianResults with per-point estimates for each spec entry
    """
    model.eval()
    model = model.to(device)
    features = features.to(device)

    if max_samples is not None and features.shape[0] > max_samples:
        idx = torch.randperm(features.shape[0])[:max_samples]
        features = features[idx]

    def forward(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    estimates: dict[str, NDArray[np.floating]] = {}
    true_values: dict[str, float] = {}
    regressions: dict[str, RegressionResult] = {}

    for spec in specs:
        tangent = torch.zeros_like(features)
        for idx in spec.perturb_indices:
            tangent[:, idx] = 1.0

        _, jvp_result = jvp(forward, (features,), (tangent,))

        coeff = (
            (
                jvp_result[:, spec.output_index] / len(spec.perturb_indices)  # type: ignore[reportCallIssue]
            )
            .detach()
            .cpu()
            .numpy()
        )

        if spec.post_extract is not None:
            raw_jvp = coeff.copy()
            feat_np = features.detach().cpu().numpy()
            coeff = spec.post_extract(raw_jvp, feat_np)
            # Compute R² of the regression
            one_minus_u = 1.0 - feat_np[:, 0]
            K_est = float(coeff[0])  # all identical from regression
            predicted = K_est * one_minus_u
            ss_res = np.sum((raw_jvp - predicted) ** 2)
            ss_tot = np.sum((raw_jvp - np.mean(raw_jvp)) ** 2)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            regressions[spec.name] = RegressionResult(
                value=K_est,
                r_squared=r_squared,
                raw_jvp=raw_jvp,
                regressor=one_minus_u,
            )

        estimates[spec.name] = coeff  # type: ignore[assignment]
        true_values[spec.name] = spec.true_value

    return JacobianResults(estimates=estimates, true_values=true_values, regressions=regressions)
