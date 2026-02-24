"""
Initial conditions for Brusselator reaction-diffusion system.

Based on literature review:
- Turing patterns emerge from small random perturbations around steady state
- The IC shape matters less than the parameter regime (k1, k2, D_u, D_v)
- Canonical approach: u = a + ε*noise, v = b/a + ε*noise

References:
- Physics Forums discussion on Brusselator Turing patterns
- Peña & Pérez-García (2001) "Stability of Turing patterns in the Brusselator model"
- PMC4006638: Simulations of pattern dynamics for reaction-diffusion systems

All ICs return (u, v, params_dict) where:
- u, v: 2D numpy arrays of shape (ny, nx)
- params_dict: Dictionary of generated parameters (for reproducibility logging)

Brusselator steady state: u* = k1, v* = k2/k1
Turing threshold: k2_c = (1 + k1 * sqrt(D_u/D_v))^2
Pattern forms when k2 > k2_c
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter


def compute_turing_threshold(k1: float, D_u: float, D_v: float) -> float:
    """
    Compute the Turing bifurcation threshold for k2.

    Pattern formation occurs when k2 > threshold.

    Args:
        k1: Reaction parameter (steady state u* = k1)
        D_u: Diffusion coefficient for u
        D_v: Diffusion coefficient for v

    Returns:
        k2_threshold: Critical value of k2 for Turing instability
    """
    return (1 + k1 * np.sqrt(D_u / D_v)) ** 2


def perturbed_uniform_ic(
    k1: float,
    k2: float,
    perturbation_amplitude: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Uniform steady state with small random perturbations.

    This is the CANONICAL IC for Brusselator pattern formation.
    Literature confirms: "small perturbations about steady state,
    any small random perturbation will do"

    Args:
        k1, k2: Reaction parameters (define steady state u*=k1, v*=k2/k1)
        perturbation_amplitude: Amplitude of random perturbations (fraction of steady state)
                               Literature uses ~0.05-0.10 (5-10%)
        x, y: Coordinate arrays
        seed: Random seed for reproducibility

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    u_star = k1
    v_star = k2 / k1

    # Create meshgrid
    X, _ = np.meshgrid(x, y)  # X, Y
    ny, nx = X.shape

    # Add small random perturbations around steady state
    # Literature form: u = a + ε * random(-1, 1)
    u = u_star + perturbation_amplitude * u_star * (2 * rng.random((ny, nx)) - 1)
    v = v_star + perturbation_amplitude * v_star * (2 * rng.random((ny, nx)) - 1)

    # Ensure non-negative
    u = np.maximum(u, 1e-6)
    v = np.maximum(v, 1e-6)

    params = {
        "type": "perturbed_uniform",
        "k1": k1,
        "k2": k2,
        "u_star": u_star,
        "v_star": v_star,
        "perturbation_amplitude": perturbation_amplitude,
        "seed": seed,
    }

    return u, v, params


def random_smooth_ic(
    k1: float,
    k2: float,
    perturbation_amplitude: float,
    smoothing_scale: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Smooth random field around steady state.

    Like perturbed_uniform but with spatially correlated noise.
    Creates smooth initial perturbations that may favor certain wavelengths.

    Args:
        k1, k2: Reaction parameters
        perturbation_amplitude: Amplitude of perturbations (fraction of steady state)
        smoothing_scale: Gaussian smoothing sigma (in grid points)
        x, y: Coordinate arrays
        seed: Random seed for reproducibility

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    u_star = k1
    v_star = k2 / k1

    # Create meshgrid
    X, _ = np.meshgrid(x, y)  # X, Y
    ny, nx = X.shape

    # Generate smooth random fields
    noise_u = rng.standard_normal((ny, nx))
    noise_v = rng.standard_normal((ny, nx))

    smooth_u = gaussian_filter(noise_u, sigma=smoothing_scale, mode="wrap")
    smooth_v = gaussian_filter(noise_v, sigma=smoothing_scale, mode="wrap")

    # Normalize to unit variance then scale
    smooth_u = smooth_u / (np.std(smooth_u) + 1e-8) * perturbation_amplitude * u_star
    smooth_v = smooth_v / (np.std(smooth_v) + 1e-8) * perturbation_amplitude * v_star

    # Add to steady state
    u = u_star + smooth_u
    v = v_star + smooth_v

    # Ensure non-negative
    u = np.maximum(u, 1e-6)
    v = np.maximum(v, 1e-6)

    params = {
        "type": "random_smooth",
        "k1": k1,
        "k2": k2,
        "u_star": u_star,
        "v_star": v_star,
        "perturbation_amplitude": perturbation_amplitude,
        "smoothing_scale": smoothing_scale,
        "seed": seed,
    }

    return u, v, params


def localized_perturbation_ic(
    k1: float,
    k2: float,
    perturbation_amplitude: float,
    patch_center: Tuple[float, float],
    patch_radius: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Localized random perturbation in a circular patch.

    Tests whether Turing patterns spread from a localized seed.
    Outside the patch: exact steady state. Inside: random perturbations.

    Args:
        k1, k2: Reaction parameters
        perturbation_amplitude: Amplitude inside patch
        patch_center: (cx, cy) center of perturbation region
        patch_radius: Radius of perturbation region
        x, y: Coordinate arrays
        seed: Random seed

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    u_star = k1
    v_star = k2 / k1

    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape
    cx, cy = patch_center

    # Start at exact steady state
    u = np.full((ny, nx), u_star)
    v = np.full((ny, nx), v_star)

    # Create mask for perturbation region
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = r < patch_radius

    # Add perturbations only inside the patch
    n_perturbed = int(np.sum(mask))
    u[mask] += perturbation_amplitude * u_star * (2 * rng.random(n_perturbed) - 1)
    v[mask] += perturbation_amplitude * v_star * (2 * rng.random(n_perturbed) - 1)

    # Ensure non-negative
    u = np.maximum(u, 1e-6)
    v = np.maximum(v, 1e-6)

    params = {
        "type": "localized_perturbation",
        "k1": k1,
        "k2": k2,
        "u_star": u_star,
        "v_star": v_star,
        "perturbation_amplitude": perturbation_amplitude,
        "patch_center": patch_center,
        "patch_radius": patch_radius,
        "seed": seed,
    }

    return u, v, params


def multi_patch_perturbation_ic(
    k1: float,
    k2: float,
    perturbation_amplitude: float,
    n_patches: int,
    patch_radius: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multiple localized perturbation patches at random locations.

    Analogous to multi_vortex in NS: tests how patterns nucleate from
    multiple spatially separated seeds and interact as they grow.

    Args:
        k1, k2: Reaction parameters
        perturbation_amplitude: Amplitude inside each patch
        n_patches: Number of circular patches to place
        patch_radius: Radius of each patch
        x, y: Coordinate arrays
        seed: Random seed

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    u_star = k1
    v_star = k2 / k1

    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape
    # Lx, Ly = x[-1] - x[0], y[-1] - y[0]

    # Start at exact steady state
    u = np.full((ny, nx), u_star)
    v = np.full((ny, nx), v_star)

    # Generate random patch centers (avoid edges)
    margin = patch_radius * 1.5
    patch_centers = []

    for _ in range(n_patches):
        cx = rng.uniform(x[0] + margin, x[-1] + (x[1] - x[0]) - margin)
        cy = rng.uniform(y[0] + margin, y[-1] + (y[1] - y[0]) - margin)
        patch_centers.append((cx, cy))

    # Apply perturbations in each patch
    for cx, cy in patch_centers:
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = r < patch_radius

        n_perturbed = int(np.sum(mask))
        u[mask] += perturbation_amplitude * u_star * (2 * rng.random(n_perturbed) - 1)
        v[mask] += perturbation_amplitude * v_star * (2 * rng.random(n_perturbed) - 1)

    # Ensure non-negative
    u = np.maximum(u, 1e-6)
    v = np.maximum(v, 1e-6)

    params = {
        "type": "multi_patch_perturbation",
        "k1": k1,
        "k2": k2,
        "u_star": u_star,
        "v_star": v_star,
        "perturbation_amplitude": perturbation_amplitude,
        "n_patches": n_patches,
        "patch_radius": patch_radius,
        "patch_centers": patch_centers,
        "seed": seed,
    }

    return u, v, params


def gradient_perturbation_ic(
    k1: float,
    k2: float,
    gradient_amplitude: float,
    n_modes: int,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Sinusoidal gradient perturbations on steady state.

    Unlike random noise ICs, this creates structured spatial variation
    with specific wavelengths. Tests whether the network can handle
    ICs that may bias toward certain pattern modes.

    The perturbation is a sum of random sinusoidal modes:
        δu = Σ a_i * sin(k_xi * x + k_yi * y + φ_i)

    Args:
        k1, k2: Reaction parameters
        gradient_amplitude: Amplitude of gradient perturbation (fraction of steady state)
        n_modes: Number of sinusoidal modes to superimpose
        x, y: Coordinate arrays
        seed: Random seed

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    u_star = k1
    v_star = k2 / k1

    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape
    Lx, Ly = x[-1] - x[0] + (x[1] - x[0]), y[-1] - y[0] + (y[1] - y[0])

    # Initialize perturbations
    delta_u = np.zeros((ny, nx))
    delta_v = np.zeros((ny, nx))

    # Generate random modes
    modes = []
    for _ in range(n_modes):
        # Random wavenumbers (1-4 wavelengths across domain)
        n_waves_x = rng.integers(1, 5)
        n_waves_y = rng.integers(1, 5)
        kx = 2 * np.pi * n_waves_x / Lx
        ky = 2 * np.pi * n_waves_y / Ly

        # Random phases and amplitudes
        phase_u = rng.uniform(0, 2 * np.pi)
        phase_v = rng.uniform(0, 2 * np.pi)
        amp_u = rng.uniform(0.5, 1.5)  # Relative amplitude variation
        amp_v = rng.uniform(0.5, 1.5)

        delta_u += amp_u * np.sin(kx * X + ky * Y + phase_u)
        delta_v += amp_v * np.sin(kx * X + ky * Y + phase_v)

        modes.append(
            {
                "n_waves_x": int(n_waves_x),
                "n_waves_y": int(n_waves_y),
                "phase_u": phase_u,
                "phase_v": phase_v,
                "amp_u": amp_u,
                "amp_v": amp_v,
            }
        )

    # Normalize and scale
    delta_u = delta_u / n_modes * gradient_amplitude * u_star
    delta_v = delta_v / n_modes * gradient_amplitude * v_star

    # Apply to steady state
    u = u_star + delta_u
    v = v_star + delta_v

    # Ensure non-negative
    u = np.maximum(u, 1e-6)
    v = np.maximum(v, 1e-6)

    params = {
        "type": "gradient_perturbation",
        "k1": k1,
        "k2": k2,
        "u_star": u_star,
        "v_star": v_star,
        "gradient_amplitude": gradient_amplitude,
        "n_modes": n_modes,
        "modes": modes,
        "seed": seed,
    }

    return u, v, params


def create_brusselator_ic(
    ic_config: dict, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Factory function to create Brusselator IC from configuration dict.

    Args:
        ic_config: Dict with 'type' and type-specific parameters
        x, y: Coordinate arrays

    Returns:
        (u, v, params_dict)
    """
    ic_type = ic_config["type"]

    # Extract k1, k2 from config (required for all ICs)
    k1 = ic_config["k1"]
    k2 = ic_config["k2"]
    seed = ic_config.get("seed", None)

    if ic_type == "perturbed_uniform":
        return perturbed_uniform_ic(
            k1=k1,
            k2=k2,
            perturbation_amplitude=ic_config.get("perturbation_amplitude", 0.05),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "random_smooth":
        return random_smooth_ic(
            k1=k1,
            k2=k2,
            perturbation_amplitude=ic_config.get("perturbation_amplitude", 0.05),
            smoothing_scale=ic_config.get("smoothing_scale", 3.0),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "localized_perturbation":
        # Default to domain center
        Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])
        default_center = (Lx / 2, Ly / 2)
        default_radius = min(Lx, Ly) / 4

        return localized_perturbation_ic(
            k1=k1,
            k2=k2,
            perturbation_amplitude=ic_config.get("perturbation_amplitude", 0.10),
            patch_center=tuple(ic_config.get("patch_center", default_center)),
            patch_radius=ic_config.get("patch_radius", default_radius),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "multi_patch_perturbation":
        # Default to 3 patches distributed across domain
        Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])
        n_patches = ic_config.get("n_patches", 3)
        default_radius = min(Lx, Ly) / 8

        return multi_patch_perturbation_ic(
            k1=k1,
            k2=k2,
            perturbation_amplitude=ic_config.get("perturbation_amplitude", 0.10),
            n_patches=n_patches,
            patch_radius=ic_config.get("patch_radius", default_radius),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "gradient_perturbation":
        return gradient_perturbation_ic(
            k1=k1,
            k2=k2,
            gradient_amplitude=ic_config.get("gradient_amplitude", 0.15),
            n_modes=ic_config.get("n_modes", 2),
            x=x,
            y=y,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown Brusselator IC type: {ic_type}")
