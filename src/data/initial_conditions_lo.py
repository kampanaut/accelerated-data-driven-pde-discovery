"""
Initial conditions for Lambda-Omega reaction-diffusion system.

Lambda-Omega in the oscillatory regime (a > 0) supports:
- Spiral waves from phase singularities
- Target patterns from localized bumps
- Self-organization from random noise (unstable origin)

Limit-cycle amplitude: R_0 = sqrt(a)
Frequency: omega = c*a - 1

All ICs return (u, v, params_dict) where:
- u, v: 2D numpy arrays of shape (ny, nx)
- params_dict: Dictionary of generated parameters (for reproducibility)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def single_spiral_ic(
    center: Tuple[float, float],
    chirality: int,
    k: float,
    r_core: float,
    a: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Phase singularity spiral wave IC.

    u = R_0 * tanh(r/r_core) * cos(m*theta - k*r)
    v = R_0 * tanh(r/r_core) * sin(m*theta - k*r)

    where R_0 = sqrt(a), m = chirality (+1 or -1).
    The tanh envelope smoothly goes to zero at the core,
    creating a topological defect that drives rotation.

    Args:
        center: (cx, cy) center of spiral
        chirality: +1 or -1 (rotation direction)
        k: radial wavenumber (controls arm spacing)
        r_core: core radius (smoothing scale)
        a: growth rate (sets amplitude R_0 = sqrt(a))
        x, y: Coordinate arrays
        seed: Random seed (unused, for interface consistency)

    Returns:
        (u, v, params_dict)
    """
    X, Y = np.meshgrid(x, y)
    cx, cy = center

    dx = X - cx
    dy = Y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    R_0 = np.sqrt(a)
    envelope = R_0 * np.tanh(r / r_core)
    phase = chirality * theta - k * r

    u = envelope * np.cos(phase)
    v = envelope * np.sin(phase)

    params = {
        "type": "single_spiral",
        "center": center,
        "chirality": chirality,
        "k": k,
        "r_core": r_core,
        "seed": seed,
    }
    return u, v, params


def random_perturbation_ic(
    amp: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Small random noise around the unstable origin (0, 0).

    For Lambda-Omega with a > 0, the origin is unstable.
    Any perturbation grows to the limit cycle and self-organizes
    into spiral waves through symmetry-breaking instabilities.

    u = amp * noise, v = amp * noise (independent smoothed noise)

    Args:
        amp: Noise amplitude
        x, y: Coordinate arrays
        seed: Random seed

    Returns:
        (u, v, params_dict)
    """
    rng = np.random.default_rng(seed)
    X, _ = np.meshgrid(x, y)
    ny, nx = X.shape

    u = amp * (2 * rng.random((ny, nx)) - 1)
    v = amp * (2 * rng.random((ny, nx)) - 1)

    params = {
        "type": "random_perturbation",
        "amp": amp,
        "seed": seed,
    }
    return u, v, params


def target_pattern_ic(
    amp: float,
    center: Tuple[float, float],
    radius: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Gaussian bump IC that emits concentric target rings.

    u = amp * exp(-r^2 / (2*sigma^2)), v = 0

    The bump acts as a pacemaker: the localized perturbation
    grows to limit-cycle amplitude and emits outward-propagating
    concentric rings (target pattern).

    Args:
        amp: Bump amplitude
        center: (cx, cy) center of bump
        radius: Gaussian sigma
        x, y: Coordinate arrays
        seed: Random seed (unused, for interface consistency)

    Returns:
        (u, v, params_dict)
    """
    X, Y = np.meshgrid(x, y)
    cx, cy = center

    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    u = amp * np.exp(-r2 / (2 * radius ** 2))
    v = np.zeros_like(u)

    params = {
        "type": "target_pattern",
        "amp": amp,
        "center": center,
        "radius": radius,
        "seed": seed,
    }
    return u, v, params


def multi_pacemaker_ic(
    amp: float,
    n_bumps: int,
    radius: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multiple Gaussian bumps at random locations.

    Each bump acts as an independent pacemaker emitting target rings.
    Rings from different sources interact, creating complex patterns.

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
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        u += amp * np.exp(-r2 / (2 * radius ** 2))

    v = np.zeros_like(u)

    params = {
        "type": "multi_pacemaker",
        "amp": amp,
        "n_bumps": n_bumps,
        "radius": radius,
        "bump_centers": bump_centers,
        "seed": seed,
    }
    return u, v, params


def invasion_patch_ic(
    patch_radius: float,
    width: float,
    center: Tuple[float, float],
    a: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Smoothed disk at limit-cycle amplitude.

    u = sqrt(a) * 0.5 * (1 - tanh((r - R) / w)), v = 0

    A patch already at the limit-cycle amplitude invades
    the surrounding quiescent region. The invasion front
    emits waves as the oscillation spreads outward.

    Args:
        patch_radius: Radius of the disk (R)
        width: Smoothing width of the tanh transition (w)
        center: (cx, cy) center of patch
        a: Growth rate (sets amplitude sqrt(a))
        x, y: Coordinate arrays
        seed: Random seed (unused, for interface consistency)

    Returns:
        (u, v, params_dict)
    """
    X, Y = np.meshgrid(x, y)
    cx, cy = center

    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    R_0 = np.sqrt(a)
    u = R_0 * 0.5 * (1.0 - np.tanh((r - patch_radius) / width))
    v = np.zeros_like(u)

    params = {
        "type": "invasion_patch",
        "patch_radius": patch_radius,
        "width": width,
        "center": center,
        "seed": seed,
    }
    return u, v, params


# =============================================================================
# OOD (test-only) IC types
# =============================================================================


def multi_arm_spiral_ic(
    center: Tuple[float, float],
    m: int,
    k: float,
    r_core: float,
    a: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Multi-arm spiral IC (m = 2, 3, ...). Unstable, breaks up.

    Same formula as single_spiral but with |m| > 1:
    u = R_0 * tanh(r/r_core) * cos(m*theta - k*r)
    v = R_0 * tanh(r/r_core) * sin(m*theta - k*r)

    Multi-arm spirals are topologically unstable and break apart
    into single-arm spirals, creating complex transient dynamics.

    Args:
        center: (cx, cy) center of spiral
        m: Number of arms (2 or 3)
        k: Radial wavenumber
        r_core: Core radius
        a: Growth rate
        x, y: Coordinate arrays
        seed: Random seed (unused, for interface consistency)

    Returns:
        (u, v, params_dict)
    """
    X, Y = np.meshgrid(x, y)
    cx, cy = center

    dx = X - cx
    dy = Y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    R_0 = np.sqrt(a)
    envelope = R_0 * np.tanh(r / r_core)
    phase = m * theta - k * r

    u = envelope * np.cos(phase)
    v = envelope * np.sin(phase)

    params = {
        "type": "multi_arm_spiral",
        "center": center,
        "m": m,
        "k": k,
        "r_core": r_core,
        "seed": seed,
    }
    return u, v, params


def plane_wave_ic(
    q: float,
    a: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Exact plane travelling wave (PTW) solution.

    u = sqrt(a - q^2) * cos(q*x)
    v = sqrt(a - q^2) * sin(q*x)

    This is an exact solution of Lambda-Omega for q^2 < a.
    The wave propagates in the x-direction with constant amplitude.

    Args:
        q: Spatial wavenumber (must satisfy q^2 < a)
        a: Growth rate
        x, y: Coordinate arrays
        seed: Random seed (unused, for interface consistency)

    Returns:
        (u, v, params_dict)
    """
    X, _ = np.meshgrid(x, y)

    amp = np.sqrt(max(a - q ** 2, 0.0))
    u = amp * np.cos(q * X)
    v = amp * np.sin(q * X)

    params = {
        "type": "plane_wave",
        "q": q,
        "seed": seed,
    }
    return u, v, params


# =============================================================================
# Factory
# =============================================================================


def create_lo_ic(
    ic_config: dict, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Factory function to create Lambda-Omega IC from configuration dict.

    Args:
        ic_config: Dict with 'type' and type-specific parameters
        x, y: Coordinate arrays

    Returns:
        (u, v, params_dict)
    """
    ic_type = ic_config["type"]
    seed = ic_config.get("seed", None)
    # a is needed by several ICs for limit-cycle amplitude
    a = ic_config.get("a_value", 1.0)

    Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])

    if ic_type == "single_spiral":
        default_center = (Lx / 2, Ly / 2)
        return single_spiral_ic(
            center=tuple(ic_config.get("center", default_center)),
            chirality=ic_config.get("chirality", 1),
            k=ic_config.get("k", 0.5),
            r_core=ic_config.get("r_core", 2.0),
            a=a,
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "random_perturbation":
        return random_perturbation_ic(
            amp=ic_config.get("amp", 0.1),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "target_pattern":
        default_center = (Lx / 2, Ly / 2)
        default_radius = min(Lx, Ly) / 10
        return target_pattern_ic(
            amp=ic_config.get("amp", 1.0),
            center=tuple(ic_config.get("center", default_center)),
            radius=ic_config.get("radius", default_radius),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "multi_pacemaker":
        default_radius = min(Lx, Ly) / 10
        return multi_pacemaker_ic(
            amp=ic_config.get("amp", 1.0),
            n_bumps=ic_config.get("n_bumps", 3),
            radius=ic_config.get("radius", default_radius),
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "invasion_patch":
        default_center = (Lx / 2, Ly / 2)
        return invasion_patch_ic(
            patch_radius=ic_config.get("patch_radius", min(Lx, Ly) / 5),
            width=ic_config.get("width", 1.0),
            center=tuple(ic_config.get("center", default_center)),
            a=a,
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "multi_arm_spiral":
        default_center = (Lx / 2, Ly / 2)
        return multi_arm_spiral_ic(
            center=tuple(ic_config.get("center", default_center)),
            m=ic_config.get("m", 2),
            k=ic_config.get("k", 0.5),
            r_core=ic_config.get("r_core", 2.0),
            a=a,
            x=x,
            y=y,
            seed=seed,
        )

    elif ic_type == "plane_wave":
        return plane_wave_ic(
            q=ic_config.get("q", 0.5),
            a=a,
            x=x,
            y=y,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown Lambda-Omega IC type: {ic_type}")
