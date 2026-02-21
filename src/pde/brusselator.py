"""
Brusselator reaction-diffusion solver using Dedalus spectral methods.

The Brusselator model:
    u_t = D_u nabla^2(u) + k1 - (k2 + 1)*u + u^2*v
    v_t = D_v nabla^2(v) + k2*u - u^2*v

where:
    u, v: concentrations
    D_u, D_v: diffusion coefficients
    k1, k2: reaction rate constants

Steady state: u* = k1, v* = k2/k1
Turing instability requires D_v > D_u and specific parameter relationships.

IMEX splitting:
    LHS (implicit): diffusion + linear reaction terms
    RHS (explicit): nonlinear reaction terms (u^2*v)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import dedalus.public as d3
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, TypedDict, Any, Union, cast


# =============================================================================
# Dataclasses for type-safe configuration
# =============================================================================


@dataclass
class BrusselatorSimParams:
    """Simulation parameters for Brusselator solver."""

    D_u: float  # Diffusion coefficient for u
    D_v: float  # Diffusion coefficient for v
    k1: float  # Reaction rate (u* = k1)
    k2: float  # Reaction rate (v* = k2/k1)
    domain_size: Tuple[float, float]  # (Lx, Ly)
    resolution: Tuple[int, int]  # (ny, nx)
    t_end: float  # Final simulation time
    dt: float  # Timestep
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
    D_u: Optional[float] = None
    D_v: Optional[float] = None


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
    patch_radius: Optional[float] = None  # Default: L/4


@dataclass
class MultiPatchPerturbationIC(BrusselatorICBase):
    """Multiple localized perturbation patches at random locations."""

    perturbation_amplitude: float = 0.10
    n_patches: int = 3
    patch_radius: Optional[float] = None  # Default: L/8


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


def solve_brusselator(
    initial_concentration: Tuple[np.ndarray, np.ndarray],
    D_u: float,
    D_v: float,
    k1: float,
    k2: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: Optional[float] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D Brusselator reaction-diffusion equations using Dedalus.

    IMEX splitting: linear terms implicit (LHS), nonlinear terms explicit (RHS).

    Args:
        initial_concentration: Tuple of (u, v) numpy arrays, shape (ny, nx)
        D_u: Diffusion coefficient for species u
        D_v: Diffusion coefficient for species v
        k1: Reaction rate constant (feed rate)
        k2: Reaction rate constant
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep for time integration
        save_interval: How often to save snapshots. If None, save every step.

    Returns:
        Tuple of:
        - concentration_history: List of (u, v) tuples at saved timesteps
        - times: Array of time values for saved snapshots
        - x: 1D array of x-coordinates
        - y: 1D array of y-coordinates
    """
    u_init, v_init = initial_concentration
    ny, nx = u_init.shape
    Lx, Ly = domain_size

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    # --- Dedalus setup ---
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
    ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=3/2)

    u = dist.Field(name='u', bases=(xbasis, ybasis))
    v = dist.Field(name='v', bases=(xbasis, ybasis))
    u['g'] = u_init
    v['g'] = v_init

    # IMEX: LHS = implicit linear, RHS = explicit nonlinear
    #   u_t = D_u*lap(u) + k1 - (k2+1)*u + u^2*v
    #   v_t = D_v*lap(v) + k2*u - u^2*v
    #
    # Rearranged:
    #   dt(u) - D_u*lap(u) + (k2+1)*u = k1 + u*u*v
    #   dt(v) - D_v*lap(v)             = k2*u - u*u*v
    problem = d3.IVP([u, v], namespace={'u': u, 'v': v, 'D_u': D_u, 'D_v': D_v, 'k1': k1, 'k2': k2})
    problem.add_equation("dt(u) - D_u*lap(u) + (k2+1)*u = k1 + u*u*v")
    problem.add_equation("dt(v) - D_v*lap(v) = k2*u - u*u*v")
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = t_end

    # --- Snapshot loop ---
    concentration_history: List[Tuple[np.ndarray, np.ndarray]] = []
    times: List[float] = []

    if save_interval is None:
        save_interval = dt

    save_every = max(1, int(save_interval / dt))
    step = 0

    # Save initial condition
    u.change_scales(1)
    v.change_scales(1)
    concentration_history.append((np.array(u['g']).copy(), np.array(v['g']).copy()))
    times.append(0.0)

    print("Starting Brusselator simulation:")
    print(f"  Domain: {Lx} x {Ly}")
    print(f"  Resolution: {nx} x {ny}")
    print(f"  Diffusion: D_u = {D_u}, D_v = {D_v}")
    print(f"  Reaction: k1 = {k1}, k2 = {k2}")
    print(f"  Steady state: u* = {k1:.4f}, v* = {(k2 / k1):.4f}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")

    while solver.proceed:
        solver.step(dt)
        step += 1

        if step % save_every == 0:
            u.change_scales(1)
            v.change_scales(1)
            concentration_history.append(
                (np.array(u['g']).copy(), np.array(v['g']).copy())
            )
            times.append(solver.sim_time)

            if step % (save_every * 10) == 0:
                u_mean = np.mean(u['g'])
                v_mean = np.mean(v['g'])
                print(
                    f"  t = {solver.sim_time:.3f} / {t_end}  |  <u> = {u_mean:.4f}, <v> = {v_mean:.4f}"
                )

    print(f"Simulation complete. Saved {len(concentration_history)} snapshots.")

    return concentration_history, np.array(times), x, y


def solve_brusselator_with_params(
    ic_params: BrusselatorICParams | dict[str, Any],
    simulation_params: BrusselatorSimParams | dict[str, Any],
) -> SolverResult:
    """
    High-level interface: generate IC from parameters and solve Brusselator.

    Args:
        ic_params: Dataclass or dict with keys:
            - 'type': IC type name
            - type-specific parameters
            - OR 'u_init', 'v_init' for custom IC
        simulation_params: Dataclass or dict with keys:
            - 'D_u': Diffusion coefficient for u
            - 'D_v': Diffusion coefficient for v
            - 'k1': Reaction rate constant
            - 'k2': Reaction rate constant
            - 'domain_size': (Lx, Ly)
            - 'resolution': (ny, nx)
            - 't_end': Final time
            - 'dt': Timestep
            - 'save_interval': Snapshot interval

    Returns:
        Dict containing:
        - 'concentration_history': List of (u, v) tuples
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
    ny, nx = sim_dict["resolution"]
    Lx, Ly = sim_dict["domain_size"]

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    # Generate initial condition
    if ic_dict.get("type") == "custom":
        u_init = ic_dict["u_init"]
        v_init = ic_dict["v_init"]
        generated_params = {}
    else:
        u_init, v_init, generated_params = create_brusselator_ic(ic_dict, x, y)

    # Solve Brusselator
    concentration_history, times, x_out, y_out = solve_brusselator(
        initial_concentration=(u_init, v_init),
        D_u=sim_dict["D_u"],
        D_v=sim_dict["D_v"],
        k1=sim_dict["k1"],
        k2=sim_dict["k2"],
        domain_size=sim_dict["domain_size"],
        t_end=sim_dict["t_end"],
        dt=sim_dict["dt"],
        save_interval=sim_dict["save_interval"],
    )

    return {
        "concentration_history": concentration_history,
        "times": times,
        "x": x_out,
        "y": y_out,
        "ic_params": ic_dict.copy(),
        "simulation_params": sim_dict.copy(),
        "generated_params": generated_params,
    }
