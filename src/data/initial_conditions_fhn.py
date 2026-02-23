"""
Initial conditions for FitzHugh-Nagumo reaction-diffusion system.

FHN in the excitable regime (eps << 1, b ~ 0) supports:
- Expanding pulses from super-threshold bumps
- Spiral waves from broken wavefronts
- Complex collision dynamics from multiple bumps

Steady state for b=0: (u*, v*) = (0, 0)

All ICs return (u, v, params_dict) where:
- u, v: 2D numpy arrays of shape (ny, nx)
- params_dict: Dictionary of generated parameters (for reproducibility)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def random_perturbation_ic(
    perturbation_amplitude: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Small random noise around the resting state (0, 0).

    Whether this triggers activity depends on parameters — small noise
    may decay in the excitable regime, or grow in the oscillatory regime.

    Args:
        perturbation_amplitude: Amplitude of uniform random noise
        x, y: Coordinate arrays
        seed: Random seed

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)
    X, _ = np.meshgrid(x, y)
    ny, nx = X.shape

    u = perturbation_amplitude * (2 * rng.random((ny, nx)) - 1)
    v = np.zeros((ny, nx))

    params = {
        "type": "random_perturbation",
        "perturbation_amplitude": perturbation_amplitude,
        "seed": seed,
    }
    return u, v, params


def localized_bump_ic(
    amp: float,
    center: Tuple[float, float],
    radius: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Gaussian bump at a specified location. Super-threshold bump
    triggers an expanding circular pulse in the excitable regime.

    u = amp * exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
    v = 0

    Args:
        amp: Bump amplitude (should exceed excitation threshold ~ 0.5-1.0)
        center: (cx, cy) center of bump
        radius: Gaussian sigma
        x, y: Coordinate arrays
        seed: Random seed (unused, kept for interface consistency)

    Returns:
        (u, v, params_dict)
    """
    X, Y = np.meshgrid(x, y)
    cx, cy = center

    u = amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * radius ** 2))
    v = np.zeros_like(u)

    params = {
        "type": "localized_bump",
        "amp": amp,
        "center": center,
        "radius": radius,
        "seed": seed,
    }
    return u, v, params


def multi_bump_ic(
    amp: float,
    n_bumps: int,
    radius: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multiple Gaussian bumps at random locations. Pulses expand,
    collide, and annihilate — creates complex transient dynamics.

    Args:
        amp: Bump amplitude
        n_bumps: Number of bumps
        radius: Gaussian sigma for each bump
        x, y: Coordinate arrays
        seed: Random seed

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape

    margin = radius * 2.0
    u = np.zeros((ny, nx))
    bump_centers = []

    for _ in range(n_bumps):
        cx = rng.uniform(x[0] + margin, x[-1] + (x[1] - x[0]) - margin)
        cy = rng.uniform(y[0] + margin, y[-1] + (y[1] - y[0]) - margin)
        bump_centers.append((cx, cy))
        u += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * radius ** 2))

    v = np.zeros_like(u)

    params = {
        "type": "multi_bump",
        "amp": amp,
        "n_bumps": n_bumps,
        "radius": radius,
        "bump_centers": bump_centers,
        "seed": seed,
    }
    return u, v, params


def broken_wavefront_ic(
    front_x: float,
    gap_y: float,
    gap_height: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Broken wavefront IC — the classic spiral wave recipe for FHN.

    A planar wavefront (u=1 behind, u=0 ahead) with a gap cut in it.
    The broken end curls into a spiral wave. Behind the front, v is set
    to a positive value to create a refractory tail.

    Args:
        front_x: x-position of the wavefront
        gap_y: y-position where the gap starts
        gap_height: height of the gap in y
        x, y: Coordinate arrays
        seed: Random seed (unused, for interface consistency)

    Returns:
        (u, v, params_dict)
    """
    X, Y = np.meshgrid(x, y)

    # Excited region behind the front
    u = np.where(X < front_x, 1.0, 0.0)

    # Cut the gap — zero out u where y is in [gap_y, gap_y + gap_height]
    gap_mask = (Y >= gap_y) & (Y <= gap_y + gap_height)
    u[gap_mask] = 0.0

    # Refractory tail behind front (speeds up spiral formation)
    v = np.where(X < front_x, 0.5, 0.0)
    v[gap_mask] = 0.0

    params = {
        "type": "broken_wavefront",
        "front_x": front_x,
        "gap_y": gap_y,
        "gap_height": gap_height,
        "seed": seed,
    }
    return u, v, params


def perturbed_front_ic(
    front_x: float,
    width: float,
    amp: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Smooth planar wavefront with noise perturbation.

    u = sigmoid((front_x - x) / width) + amp * noise
    v = 0

    The sigmoid creates a smooth transition from excited to resting.
    Noise perturbation can develop into wavefront instabilities.

    Args:
        front_x: x-position of wavefront center
        width: Sigmoid transition width
        amp: Noise amplitude
        x, y: Coordinate arrays
        seed: Random seed

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)
    X, _ = np.meshgrid(x, y)
    ny, nx = X.shape

    # Smooth wavefront via sigmoid
    u = 1.0 / (1.0 + np.exp((X - front_x) / width))
    # Add noise
    u = u + amp * (2 * rng.random((ny, nx)) - 1)
    v = np.zeros((ny, nx))

    params = {
        "type": "perturbed_front",
        "front_x": front_x,
        "width": width,
        "amp": amp,
        "seed": seed,
    }
    return u, v, params


def create_fhn_ic(
    ic_config: dict, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Factory function to create FHN IC from configuration dict.

    Args:
        ic_config: Dict with 'type' and type-specific parameters
        x, y: Coordinate arrays

    Returns:
        (u, v, params_dict)
    """
    ic_type = ic_config["type"]
    seed = ic_config.get("seed", None)

    if ic_type == "random_perturbation":
        return random_perturbation_ic(
            perturbation_amplitude=ic_config.get("perturbation_amplitude", 0.1),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "localized_bump":
        Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])
        default_center = (Lx / 2, Ly / 2)
        default_radius = min(Lx, Ly) / 10

        return localized_bump_ic(
            amp=ic_config.get("amp", 1.0),
            center=tuple(ic_config.get("center", default_center)),
            radius=ic_config.get("radius", default_radius),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "multi_bump":
        Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])
        default_radius = min(Lx, Ly) / 10

        return multi_bump_ic(
            amp=ic_config.get("amp", 1.0),
            n_bumps=ic_config.get("n_bumps", 3),
            radius=ic_config.get("radius", default_radius),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "broken_wavefront":
        Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])

        return broken_wavefront_ic(
            front_x=ic_config.get("front_x", Lx / 3),
            gap_y=ic_config.get("gap_y", Ly * 0.4),
            gap_height=ic_config.get("gap_height", Ly * 0.2),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "perturbed_front":
        Lx = x[-1] + (x[1] - x[0])

        return perturbed_front_ic(
            front_x=ic_config.get("front_x", Lx / 3),
            width=ic_config.get("width", 0.5),
            amp=ic_config.get("amp", 0.05),
            x=x,
            y=y,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown FHN IC type: {ic_type}")
