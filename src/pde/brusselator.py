"""
Brusselator reaction-diffusion solver using PhiFlow.

The Brusselator model:
    ∂A/∂t = D_A ∇²A + k₁ - (k₂ + 1)A + A²B
    ∂B/∂t = D_B ∇²B + k₂A - A²B

where:
    A, B: concentrations
    D_A, D_B: diffusion coefficients
    k₁, k₂: reaction rate constants

Steady state: A = k₁, B = k₂/k₁
Turing instability requires D_B > D_A and specific parameter relationships.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, TypedDict, Any, Union, cast

from phi.jax.flow import *  # noqa: F401,F403 — activates JAX backend for PhiFlow
from phi.field import CenteredGrid
from phi.geom import Box
from phi.math import spatial, extrapolation
from phi.physics import diffuse
from phiml.math._optimize import Solve
from phi import math


# =============================================================================
# Dataclasses for type-safe configuration
# =============================================================================

@dataclass
class BrusselatorSimParams:
    """Simulation parameters for Brusselator solver."""
    D_A: float                           # Diffusion coefficient for A
    D_B: float                           # Diffusion coefficient for B
    k1: float                            # Reaction rate (A* = k1)
    k2: float                            # Reaction rate (B* = k2/k1)
    domain_size: Tuple[float, float]     # (Lx, Ly)
    resolution: Tuple[int, int]          # (ny, nx)
    t_end: float                         # Final simulation time
    dt: float                            # Timestep
    save_interval: Optional[float] = None  # Snapshot interval (None = every step)


@dataclass
class BrusselatorICBase:
    """Base class for Brusselator IC parameters."""
    type: str
    name: str
    k1: float
    k2: float
    seed: Optional[int] = None
    # Optional overrides for simulation params
    D_A: Optional[float] = None
    D_B: Optional[float] = None


@dataclass
class PerturbedUniformIC(BrusselatorICBase):
    """Uniform steady state with small random perturbations."""
    perturbation_amplitude: float = 0.05


@dataclass
class RandomSmoothIC(BrusselatorICBase):
    """Smooth random field around steady state."""
    perturbation_amplitude: float = 0.05
    smoothing_scale: float = 3.0


@dataclass
class LocalizedPerturbationIC(BrusselatorICBase):
    """Localized random perturbation in a circular patch."""
    perturbation_amplitude: float = 0.10
    patch_center: Optional[Tuple[float, float]] = None  # Default: domain center
    patch_radius: Optional[float] = None                 # Default: L/4


@dataclass
class MultiPatchPerturbationIC(BrusselatorICBase):
    """Multiple localized perturbation patches at random locations."""
    perturbation_amplitude: float = 0.10
    n_patches: int = 3
    patch_radius: Optional[float] = None                 # Default: L/8


@dataclass
class GradientPerturbationIC(BrusselatorICBase):
    """Sinusoidal gradient perturbations on steady state."""
    gradient_amplitude: float = 0.15
    n_modes: int = 2


# Union type for all IC parameter types
BrusselatorICParams = Union[
    PerturbedUniformIC,
    RandomSmoothIC,
    LocalizedPerturbationIC,
    MultiPatchPerturbationIC,
    GradientPerturbationIC,
]


# =============================================================================
# TypedDict for solver results
# =============================================================================

class SolverResult(TypedDict):
    """Result from solve_brusselator_with_params."""
    concentration_history: List[Tuple[np.ndarray, np.ndarray]]
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ic_params: dict[str, Any]
    simulation_params: dict[str, Any]
    generated_params: dict[str, Any]


def brusselator_reaction(A, B, k1: float, k2: float):
    """
    Compute reaction terms for Brusselator.

    R_A = k₁ - (k₂ + 1)A + A²B
    R_B = k₂A - A²B

    Args:
        A, B: Concentration fields (numpy or JAX arrays)
        k1, k2: Reaction rate constants

    Returns:
        (R_A, R_B): Reaction terms for each species (same type as input)
    """
    A_squared_B = A * A * B
    R_A = k1 - (k2 + 1) * A + A_squared_B
    R_B = k2 * A - A_squared_B
    return R_A, R_B


def solve_brusselator(
    initial_concentration: Tuple[np.ndarray, np.ndarray],
    D_A: float,
    D_B: float,
    k1: float,
    k2: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: Optional[float] = None
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D Brusselator reaction-diffusion equations using PhiFlow.

    Uses operator splitting: diffusion (implicit, unconditionally stable) then reaction (explicit Euler).

    Args:
        initial_concentration: Tuple of (A, B) numpy arrays, shape (ny, nx)
        D_A: Diffusion coefficient for species A
        D_B: Diffusion coefficient for species B
        k1: Reaction rate constant (feed rate)
        k2: Reaction rate constant
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep for time integration
        save_interval: How often to save snapshots. If None, save every step.

    Returns:
        Tuple of:
        - concentration_history: List of (A, B) tuples at saved timesteps
        - times: Array of time values for saved snapshots
        - x: 1D array of x-coordinates
        - y: 1D array of y-coordinates
    """
    A_init, B_init = initial_concentration
    ny, nx = A_init.shape
    Lx, Ly = domain_size

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Create domain box (PhiFlow uses **kwargs for dimension names)
    domain = Box(x=(0, Lx), y=(0, Ly))  # type: ignore[call-arg]

    # Initialize concentration fields as CenteredGrids
    A_field = math.tensor(A_init, spatial('y,x'))
    B_field = math.tensor(B_init, spatial('y,x'))

    A = CenteredGrid(A_field, extrapolation.PERIODIC, bounds=domain)
    B = CenteredGrid(B_field, extrapolation.PERIODIC, bounds=domain)

    # Initialize storage
    concentration_history = []
    times = []

    # Determine save frequency
    if save_interval is None:
        save_interval = dt

    n_steps = int(t_end / dt)
    save_every = max(1, int(save_interval / dt))

    # Save initial condition
    A_data = math.reshaped_native(A.values, ['y', 'x'])
    B_data = math.reshaped_native(B.values, ['y', 'x'])
    concentration_history.append((np.array(A_data), np.array(B_data)))
    times.append(0.0)

    print("Starting Brusselator simulation:")
    print(f"  Domain: {Lx} × {Ly}")
    print(f"  Resolution: {nx} × {ny}")
    print(f"  Diffusion: D_A = {D_A}, D_B = {D_B}")
    print(f"  Reaction: k₁ = {k1}, k₂ = {k2}")
    print(f"  Steady state: A* = {k1:.4f}, B* = {k2/k1:.4f}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")
    print(f"  Total steps: {n_steps}, saving every {save_every} steps")

    # Time integration loop (Strang splitting — O(dt²) splitting error)
    for step in range(1, n_steps + 1):
        t = step * dt

        # Step 1: Half-step reaction (explicit Euler, dt/2)
        A_val = A.values
        B_val = B.values
        R_A, R_B = brusselator_reaction(A_val, B_val, k1, k2)
        A_val = math.maximum(A_val + (dt / 2) * R_A, 0.0)
        B_val = math.maximum(B_val + (dt / 2) * R_B, 0.0)
        A = CenteredGrid(A_val, extrapolation.PERIODIC, bounds=domain)
        B = CenteredGrid(B_val, extrapolation.PERIODIC, bounds=domain)

        # Step 2: Full-step diffusion (implicit, unconditionally stable)
        A = diffuse.implicit(A, D_A, dt, solve=Solve('CG', 1e-5, x0=None))
        B = diffuse.implicit(B, D_B, dt, solve=Solve('CG', 1e-5, x0=None))

        # Step 3: Half-step reaction (explicit Euler, dt/2)
        A_val = A.values
        B_val = B.values
        R_A, R_B = brusselator_reaction(A_val, B_val, k1, k2)
        A_val = math.maximum(A_val + (dt / 2) * R_A, 0.0)
        B_val = math.maximum(B_val + (dt / 2) * R_B, 0.0)
        A = CenteredGrid(A_val, extrapolation.PERIODIC, bounds=domain)
        B = CenteredGrid(B_val, extrapolation.PERIODIC, bounds=domain)

        # Save snapshot at specified intervals (only transfer to CPU here)
        if step % save_every == 0:
            A_data = math.reshaped_native(A.values, ['y', 'x'])
            B_data = math.reshaped_native(B.values, ['y', 'x'])
            concentration_history.append((np.array(A_data), np.array(B_data)))
            times.append(t)

            if step % (save_every * 10) == 0:  # Progress update
                A_mean = float(math.mean(A_val))
                B_mean = float(math.mean(B_val))
                print(f"  t = {t:.3f} / {t_end}  |  <A> = {A_mean:.4f}, <B> = {B_mean:.4f}")

    print(f"Simulation complete. Saved {len(concentration_history)} snapshots.")

    return concentration_history, np.array(times), x, y


def solve_brusselator_with_params(
    ic_params: BrusselatorICParams | dict[str, Any],
    simulation_params: BrusselatorSimParams | dict[str, Any]
) -> SolverResult:
    """
    High-level interface: generate IC from parameters and solve Brusselator.

    Args:
        ic_params: Dataclass or dict with keys:
            - 'type': IC type name
            - type-specific parameters
            - OR 'A_init', 'B_init' for custom IC
        simulation_params: Dataclass or dict with keys:
            - 'D_A': Diffusion coefficient for A
            - 'D_B': Diffusion coefficient for B
            - 'k1': Reaction rate constant
            - 'k2': Reaction rate constant
            - 'domain_size': (Lx, Ly)
            - 'resolution': (ny, nx)
            - 't_end': Final time
            - 'dt': Timestep
            - 'save_interval': Snapshot interval

    Returns:
        Dict containing:
        - 'concentration_history': List of (A, B) tuples
        - 'times': Time values
        - 'x', 'y': Coordinate arrays
        - 'ic_params': Copy of IC parameters
        - 'simulation_params': Copy of simulation parameters
    """
    # Import Brusselator ICs
    from src.data.initial_conditions_brusselator import create_brusselator_ic

    # Convert dataclasses to dicts for backward compatibility
    ic_dict: dict[str, Any]
    sim_dict: dict[str, Any]
    if hasattr(ic_params, '__dataclass_fields__'):
        ic_dict = asdict(cast(BrusselatorICBase, ic_params))
    else:
        ic_dict = cast(dict[str, Any], ic_params)
    if hasattr(simulation_params, '__dataclass_fields__'):
        sim_dict = asdict(cast(BrusselatorSimParams, simulation_params))
    else:
        sim_dict = cast(dict[str, Any], simulation_params)

    # Extract parameters
    ny, nx = sim_dict['resolution']
    Lx, Ly = sim_dict['domain_size']

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Generate initial condition
    if ic_dict.get('type') == 'custom':
        A_init = ic_dict['A_init']
        B_init = ic_dict['B_init']
        generated_params = {}
    else:
        A_init, B_init, generated_params = create_brusselator_ic(ic_dict, x, y)

    # Solve Brusselator
    concentration_history, times, x_out, y_out = solve_brusselator(
        initial_concentration=(A_init, B_init),
        D_A=sim_dict['D_A'],
        D_B=sim_dict['D_B'],
        k1=sim_dict['k1'],
        k2=sim_dict['k2'],
        domain_size=sim_dict['domain_size'],
        t_end=sim_dict['t_end'],
        dt=sim_dict['dt'],
        save_interval=sim_dict['save_interval']
    )

    return {
        'concentration_history': concentration_history,
        'times': times,
        'x': x_out,
        'y': y_out,
        'ic_params': ic_dict.copy(),
        'simulation_params': sim_dict.copy(),
        'generated_params': generated_params
    }
