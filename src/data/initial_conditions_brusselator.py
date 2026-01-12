"""
Initial conditions for Brusselator reaction-diffusion system.

Based on literature review:
- Turing patterns emerge from small random perturbations around steady state
- The IC shape matters less than the parameter regime (k1, k2, D_A, D_B)
- Canonical approach: u = a + ε*noise, v = b/a + ε*noise

References:
- Physics Forums discussion on Brusselator Turing patterns
- Peña & Pérez-García (2001) "Stability of Turing patterns in the Brusselator model"
- PMC4006638: Simulations of pattern dynamics for reaction-diffusion systems

All ICs return (A, B, params_dict) where:
- A, B: 2D numpy arrays of shape (ny, nx)
- params_dict: Dictionary of generated parameters (for reproducibility logging)

Brusselator steady state: A* = k1, B* = k2/k1
Turing threshold: k2_c = (1 + k1 * sqrt(D_A/D_B))^2
Pattern forms when k2 > k2_c
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter


def compute_turing_threshold(k1: float, D_A: float, D_B: float) -> float:
    """
    Compute the Turing bifurcation threshold for k2.

    Pattern formation occurs when k2 > threshold.

    Args:
        k1: Reaction parameter (steady state A* = k1)
        D_A: Diffusion coefficient for A
        D_B: Diffusion coefficient for B

    Returns:
        k2_threshold: Critical value of k2 for Turing instability
    """
    return (1 + k1 * np.sqrt(D_A / D_B)) ** 2


def perturbed_uniform_ic(
    k1: float,
    k2: float,
    perturbation_amplitude: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Uniform steady state with small random perturbations.

    This is the CANONICAL IC for Brusselator pattern formation.
    Literature confirms: "small perturbations about steady state,
    any small random perturbation will do"

    Args:
        k1, k2: Reaction parameters (define steady state A*=k1, B*=k2/k1)
        perturbation_amplitude: Amplitude of random perturbations (fraction of steady state)
                               Literature uses ~0.05-0.10 (5-10%)
        x, y: Coordinate arrays
        seed: Random seed for reproducibility

    Returns:
        (A, B, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    A_star = k1
    B_star = k2 / k1

    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape

    # Add small random perturbations around steady state
    # Literature form: u = a + ε * random(-1, 1)
    A = A_star + perturbation_amplitude * A_star * (2 * rng.random((ny, nx)) - 1)
    B = B_star + perturbation_amplitude * B_star * (2 * rng.random((ny, nx)) - 1)

    # Ensure non-negative
    A = np.maximum(A, 1e-6)
    B = np.maximum(B, 1e-6)

    params = {
        'type': 'perturbed_uniform',
        'k1': k1,
        'k2': k2,
        'A_star': A_star,
        'B_star': B_star,
        'perturbation_amplitude': perturbation_amplitude,
        'seed': seed
    }

    return A, B, params


def random_smooth_ic(
    k1: float,
    k2: float,
    perturbation_amplitude: float,
    smoothing_scale: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None
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
        (A, B, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    A_star = k1
    B_star = k2 / k1

    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape

    # Generate smooth random fields
    noise_A = rng.standard_normal((ny, nx))
    noise_B = rng.standard_normal((ny, nx))

    smooth_A = gaussian_filter(noise_A, sigma=smoothing_scale, mode='wrap')
    smooth_B = gaussian_filter(noise_B, sigma=smoothing_scale, mode='wrap')

    # Normalize to unit variance then scale
    smooth_A = smooth_A / (np.std(smooth_A) + 1e-8) * perturbation_amplitude * A_star
    smooth_B = smooth_B / (np.std(smooth_B) + 1e-8) * perturbation_amplitude * B_star

    # Add to steady state
    A = A_star + smooth_A
    B = B_star + smooth_B

    # Ensure non-negative
    A = np.maximum(A, 1e-6)
    B = np.maximum(B, 1e-6)

    params = {
        'type': 'random_smooth',
        'k1': k1,
        'k2': k2,
        'A_star': A_star,
        'B_star': B_star,
        'perturbation_amplitude': perturbation_amplitude,
        'smoothing_scale': smoothing_scale,
        'seed': seed
    }

    return A, B, params


def localized_perturbation_ic(
    k1: float,
    k2: float,
    perturbation_amplitude: float,
    patch_center: Tuple[float, float],
    patch_radius: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None
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
        (A, B, params_dict)
    """
    rng = np.random.default_rng(seed)

    # Steady state values
    A_star = k1
    B_star = k2 / k1

    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape
    cx, cy = patch_center

    # Start at exact steady state
    A = np.full((ny, nx), A_star)
    B = np.full((ny, nx), B_star)

    # Create mask for perturbation region
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = r < patch_radius

    # Add perturbations only inside the patch
    n_perturbed = np.sum(mask)
    A[mask] += perturbation_amplitude * A_star * (2 * rng.random(n_perturbed) - 1)
    B[mask] += perturbation_amplitude * B_star * (2 * rng.random(n_perturbed) - 1)

    # Ensure non-negative
    A = np.maximum(A, 1e-6)
    B = np.maximum(B, 1e-6)

    params = {
        'type': 'localized_perturbation',
        'k1': k1,
        'k2': k2,
        'A_star': A_star,
        'B_star': B_star,
        'perturbation_amplitude': perturbation_amplitude,
        'patch_center': patch_center,
        'patch_radius': patch_radius,
        'seed': seed
    }

    return A, B, params


def create_brusselator_ic(ic_config: dict, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Factory function to create Brusselator IC from configuration dict.

    Args:
        ic_config: Dict with 'type' and type-specific parameters
        x, y: Coordinate arrays

    Returns:
        (A, B, params_dict)
    """
    ic_type = ic_config['type']

    # Extract k1, k2 from config (required for all ICs)
    k1 = ic_config['k1']
    k2 = ic_config['k2']
    seed = ic_config.get('seed', None)

    if ic_type == 'perturbed_uniform':
        return perturbed_uniform_ic(
            k1=k1,
            k2=k2,
            perturbation_amplitude=ic_config.get('perturbation_amplitude', 0.05),
            x=x,
            y=y,
            seed=seed
        )

    elif ic_type == 'random_smooth':
        return random_smooth_ic(
            k1=k1,
            k2=k2,
            perturbation_amplitude=ic_config.get('perturbation_amplitude', 0.05),
            smoothing_scale=ic_config.get('smoothing_scale', 3.0),
            x=x,
            y=y,
            seed=seed
        )

    elif ic_type == 'localized_perturbation':
        # Default to domain center
        Lx, Ly = x[-1], y[-1]
        default_center = (Lx / 2, Ly / 2)
        default_radius = min(Lx, Ly) / 4

        return localized_perturbation_ic(
            k1=k1,
            k2=k2,
            perturbation_amplitude=ic_config.get('perturbation_amplitude', 0.10),
            patch_center=tuple(ic_config.get('patch_center', default_center)),
            patch_radius=ic_config.get('patch_radius', default_radius),
            x=x,
            y=y,
            seed=seed
        )

    else:
        raise ValueError(f"Unknown Brusselator IC type: {ic_type}")
