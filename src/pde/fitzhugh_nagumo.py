"""
FitzHugh-Nagumo reaction-diffusion solver using PhiFlow.

The FitzHugh-Nagumo model:
    u_t = D_u * nabla^2(u) + u - u^3 - v           (activator, fast)
    v_t = D_v * nabla^2(v) + eps * (u - a*v - b)    (recovery, slow)

where:
    u: activator (membrane potential analogue)
    v: recovery variable
    D_u, D_v: diffusion coefficients
    eps: timescale separation (small => slow recovery)
    a: recovery coupling strength
    b: excitability threshold

Steady state for b=0: (u*, v*) = (0, 0)
Excitable regime produces spiral waves with broadband spatial spectra,
structurally different from Brusselator's Turing patterns.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Any, cast

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
class FHNSimParams:
    """Simulation parameters for FitzHugh-Nagumo solver."""

    D_u: float  # Diffusion coefficient for activator
    D_v: float  # Diffusion coefficient for recovery
    eps: float  # Timescale separation
    a: float    # Recovery coupling
    b: float    # Excitability threshold
    domain_size: Tuple[float, float]  # (Lx, Ly)
    resolution: Tuple[int, int]  # (ny, nx)
    t_end: float  # Final simulation time
    dt: float  # Timestep
    save_interval: Optional[float] = None


# =============================================================================
# Reaction function
# =============================================================================


def fhn_reaction(u, v, eps: float, a: float, b: float):
    """
    Compute reaction terms for FitzHugh-Nagumo.

    R_u = u - u^3 - v
    R_v = eps * (u - a*v - b)

    Args:
        u, v: Field values (PhiFlow tensors or numpy arrays)
        eps: Timescale separation
        a: Recovery coupling
        b: Excitability threshold

    Returns:
        (R_u, R_v): Reaction terms
    """
    R_u = u - u * u * u - v
    R_v = eps * (u - a * v - b)
    return R_u, R_v


# =============================================================================
# Solver
# =============================================================================


def solve_fhn(
    initial_fields: Tuple[np.ndarray, np.ndarray],
    D_u: float,
    D_v: float,
    eps: float,
    a: float,
    b: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: Optional[float] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D FitzHugh-Nagumo reaction-diffusion equations using PhiFlow.

    Uses Strang splitting: half-reaction -> full diffusion -> half-reaction.

    Args:
        initial_fields: Tuple of (u, v) numpy arrays, shape (ny, nx)
        D_u: Diffusion coefficient for activator
        D_v: Diffusion coefficient for recovery
        eps: Timescale separation
        a: Recovery coupling
        b: Excitability threshold
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep
        save_interval: Snapshot interval (None = every step)

    Returns:
        Tuple of:
        - field_history: List of (u, v) tuples at saved timesteps
        - times: Array of time values
        - x: 1D array of x-coordinates
        - y: 1D array of y-coordinates
    """
    u_init, v_init = initial_fields
    ny, nx = u_init.shape
    Lx, Ly = domain_size

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    domain = Box(x=(0, Lx), y=(0, Ly))  # type: ignore[call-arg]

    u_field = math.tensor(u_init, spatial("y,x"))
    v_field = math.tensor(v_init, spatial("y,x"))

    u = CenteredGrid(u_field, extrapolation.PERIODIC, bounds=domain)
    v = CenteredGrid(v_field, extrapolation.PERIODIC, bounds=domain)

    field_history = []
    times = []

    if save_interval is None:
        save_interval = dt

    n_steps = int(t_end / dt)
    save_every = max(1, int(save_interval / dt))

    # Save initial condition
    u_data = math.reshaped_native(u.values, ["y", "x"])
    v_data = math.reshaped_native(v.values, ["y", "x"])
    field_history.append((np.array(u_data), np.array(v_data)))
    times.append(0.0)

    print("Starting FitzHugh-Nagumo simulation:")
    print(f"  Domain: {Lx} x {Ly}")
    print(f"  Resolution: {nx} x {ny}")
    print(f"  Diffusion: D_u = {D_u}, D_v = {D_v}")
    print(f"  Kinetics: eps = {eps}, a = {a}, b = {b}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")
    print(f"  Total steps: {n_steps}, saving every {save_every} steps")

    # Time integration loop (Strang splitting)
    for step in range(1, n_steps + 1):
        t = step * dt

        # Step 1: Half-step reaction (explicit Euler, dt/2)
        u_val = u.values
        v_val = v.values
        R_u, R_v = fhn_reaction(u_val, v_val, eps, a, b)
        u_val = u_val + (dt / 2) * R_u
        v_val = v_val + (dt / 2) * R_v
        u = CenteredGrid(u_val, extrapolation.PERIODIC, bounds=domain)
        v = CenteredGrid(v_val, extrapolation.PERIODIC, bounds=domain)

        # Step 2: Full-step diffusion (implicit, unconditionally stable)
        u = diffuse.implicit(u, D_u, dt, solve=Solve("CG", 1e-5, x0=None))
        v = diffuse.implicit(v, D_v, dt, solve=Solve("CG", 1e-5, x0=None))

        # Step 3: Half-step reaction (explicit Euler, dt/2)
        u_val = u.values
        v_val = v.values
        R_u, R_v = fhn_reaction(u_val, v_val, eps, a, b)
        u_val = u_val + (dt / 2) * R_u
        v_val = v_val + (dt / 2) * R_v
        u = CenteredGrid(u_val, extrapolation.PERIODIC, bounds=domain)
        v = CenteredGrid(v_val, extrapolation.PERIODIC, bounds=domain)

        # Save snapshot
        if step % save_every == 0:
            u_data = math.reshaped_native(u.values, ["y", "x"])
            v_data = math.reshaped_native(v.values, ["y", "x"])
            field_history.append((np.array(u_data), np.array(v_data)))
            times.append(t)

            if step % (save_every * 10) == 0:
                u_mean = float(math.mean(u_val))
                v_mean = float(math.mean(v_val))
                print(
                    f"  t = {t:.3f} / {t_end}  |  <u> = {u_mean:.4f}, <v> = {v_mean:.4f}"
                )

    print(f"Simulation complete. Saved {len(field_history)} snapshots.")

    return field_history, np.array(times), x, y


def solve_fhn_with_params(
    ic_params: dict[str, Any],
    simulation_params: FHNSimParams | dict[str, Any],
) -> dict[str, Any]:
    """
    High-level interface: generate IC from parameters and solve FHN.

    Args:
        ic_params: Dict with 'type' and type-specific parameters,
                   OR 'u_init', 'v_init' for custom IC
        simulation_params: FHNSimParams dataclass or dict

    Returns:
        Dict with field_history, times, x, y, ic_params, simulation_params
    """
    from src.data.initial_conditions_fhn import create_fhn_ic

    sim_dict: dict[str, Any]
    if hasattr(simulation_params, '__dataclass_fields__'):
        sim_dict = asdict(cast(FHNSimParams, simulation_params))
    else:
        sim_dict = cast(dict[str, Any], simulation_params)

    ny, nx = sim_dict["resolution"]
    Lx, Ly = sim_dict["domain_size"]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    generated_params: dict[str, Any]
    if ic_params.get("type") == "custom":
        u_init = ic_params["u_init"]
        v_init = ic_params["v_init"]
        generated_params = {}
    else:
        u_init, v_init, generated_params = create_fhn_ic(ic_params, x, y)

    field_history, times, x_out, y_out = solve_fhn(
        initial_fields=(u_init, v_init),
        D_u=sim_dict["D_u"],
        D_v=sim_dict["D_v"],
        eps=sim_dict["eps"],
        a=sim_dict["a"],
        b=sim_dict["b"],
        domain_size=sim_dict["domain_size"],
        t_end=sim_dict["t_end"],
        dt=sim_dict["dt"],
        save_interval=sim_dict["save_interval"],
    )

    return {
        "field_history": field_history,
        "times": times,
        "x": x_out,
        "y": y_out,
        "ic_params": ic_params.copy(),
        "simulation_params": sim_dict.copy(),
        "generated_params": generated_params,
    }
