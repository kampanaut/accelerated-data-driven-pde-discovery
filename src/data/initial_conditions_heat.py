"""
Initial conditions for the heat equation.

Since the heat equation just diffuses, any interesting spatial pattern works.
The goal is variety — different spatial scales, different numbers of peaks,
different smoothness — so the meta-learner sees diverse tasks.

All ICs return (u, params_dict) — single field, not a tuple of two.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def gaussian_bump(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    sigma: float,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Single Gaussian bump centered at (x0, y0)."""
    X, Y = np.meshgrid(x, y)
    Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])

    # Periodic distance
    dx = np.minimum(np.abs(X - x0), Lx - np.abs(X - x0))
    dy = np.minimum(np.abs(Y - y0), Ly - np.abs(Y - y0))

    u = amplitude * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    params = {
        "type": "gaussian_bump",
        "x0": x0,
        "y0": y0,
        "sigma": sigma,
        "amplitude": amplitude,
        "seed": seed,
    }
    return u, params


def multi_bump(
    x: np.ndarray,
    y: np.ndarray,
    n_bumps: int = 3,
    sigma_range: Tuple[float, float] = (0.2, 0.8),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Multiple Gaussian bumps at random positions with random sizes."""
    rng = np.random.default_rng(seed)
    Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    u = np.zeros_like(X)
    for _ in range(n_bumps):
        x0 = rng.uniform(0, Lx)
        y0 = rng.uniform(0, Ly)
        sigma = rng.uniform(sigma_range[0], sigma_range[1])
        amplitude = rng.uniform(0.5, 2.0)

        dx = np.minimum(np.abs(X - x0), Lx - np.abs(X - x0))
        dy = np.minimum(np.abs(Y - y0), Ly - np.abs(Y - y0))
        u += amplitude * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    params = {
        "type": "multi_bump",
        "n_bumps": n_bumps,
        "sigma_range": sigma_range,
        "seed": seed,
    }
    return u, params


def random_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float = 0.5,
    n_modes: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Random superposition of Fourier modes — broadband initial condition."""
    rng = np.random.default_rng(seed)
    Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    u = np.zeros_like(X)
    for _ in range(n_modes):
        kx = rng.integers(-5, 6) * 2 * np.pi / Lx
        ky = rng.integers(-5, 6) * 2 * np.pi / Ly
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.1, amplitude)
        u += amp * np.cos(kx * X + ky * Y + phase)

    params = {
        "type": "random_perturbation",
        "amplitude": amplitude,
        "n_modes": n_modes,
        "seed": seed,
    }
    return u, params


def sine_superposition(
    x: np.ndarray,
    y: np.ndarray,
    modes: int = 3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Clean superposition of a few sine modes — structured initial condition."""
    rng = np.random.default_rng(seed)
    Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])
    X, Y = np.meshgrid(x, y)

    u = np.zeros_like(X)
    for n in range(1, modes + 1):
        amp = rng.uniform(0.3, 1.5)
        u += amp * np.sin(2 * np.pi * n * X / Lx) * np.cos(2 * np.pi * n * Y / Ly)

    params = {
        "type": "sine_superposition",
        "modes": modes,
        "seed": seed,
    }
    return u, params


def create_heat_ic(
    ic_config: dict, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, Dict]:
    """
    Factory function: dispatch on ic_config["type"] to create a heat equation IC.

    Args:
        ic_config: Dict with 'type' key and type-specific parameters
        x, y: 1D coordinate arrays

    Returns:
        (u, params_dict): initial field and parameter record
    """
    ic_type = ic_config["type"]
    seed = ic_config.get("seed", None)
    Lx, Ly = x[-1] + (x[1] - x[0]), y[-1] + (y[1] - y[0])

    if ic_type == "gaussian_bump":
        u, params = gaussian_bump(
            x=x,
            y=y,
            x0=ic_config.get("x0", Lx / 2),
            y0=ic_config.get("y0", Ly / 2),
            sigma=ic_config.get("sigma", 0.5),
            amplitude=ic_config.get("amplitude", 1.0),
            seed=seed,
        )
    elif ic_type == "multi_bump":
        u, params = multi_bump(
            x=x,
            y=y,
            n_bumps=ic_config.get("n_bumps", 3),
            sigma_range=tuple(ic_config.get("sigma_range", [0.2, 0.8])),
            seed=seed,
        )
    elif ic_type == "random_perturbation":
        u, params = random_perturbation(
            x=x,
            y=y,
            amplitude=ic_config.get("amplitude", 0.5),
            n_modes=ic_config.get("n_modes", 10),
            seed=seed,
        )
    elif ic_type == "sine_superposition":
        u, params = sine_superposition(
            x=x,
            y=y,
            modes=ic_config.get("modes", 3),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown heat IC type: {ic_type}")

    # Clamp to [0, 1) — required for nonlinear heat where (1-u) must stay positive
    np.clip(u, 0.0, 1.0 - 1e-6, out=u)

    return u, params
