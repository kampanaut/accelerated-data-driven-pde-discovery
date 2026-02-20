"""
Nonlinear heat equation solver using PhiFlow.

The nonlinear heat equation:
    u_t = K * (1 - u) * nabla^2(u)

where:
    u: scalar field
    K: diffusion coefficient

The (1 - u) factor makes diffusion state-dependent:
    - Where u is small (u << 1), diffusion is nearly K (like regular heat equation)
    - Where u approaches 1, diffusion shuts off
    - Where u > 1, diffusion reverses sign (anti-diffusion, can cause instabilities)

Chris called this "deceptively hard" — same simplicity as the heat equation
but the nonlinearity creates richer dynamics.

Solver uses Strang splitting:
    1. Half-step: apply the nonlinear correction factor (1 - u) as a "reaction"
    2. Full-step: linear diffusion with coefficient K (implicit, stable)
    3. Half-step: apply the nonlinear correction factor again

More precisely, we split u_t = K*(1-u)*nabla^2(u) as:
    - Diffusion part: u_t = K * nabla^2(u)
    - Correction part: multiply the Laplacian contribution by (1-u)

This is handled by doing linear diffusion then correcting, rather than
a clean reaction/diffusion split. An alternative is to use small explicit
timesteps for the full nonlinear equation.
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


@dataclass
class NLHeatSimParams:
    """Simulation parameters for nonlinear heat equation solver."""

    K: float  # Diffusion coefficient
    domain_size: Tuple[float, float]  # (Lx, Ly)
    resolution: Tuple[int, int]  # (ny, nx)
    t_end: float  # Final simulation time
    dt: float  # Timestep
    save_interval: Optional[float] = None


def solve_nl_heat(
    initial_field: np.ndarray,
    K: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: Optional[float] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D nonlinear heat equation u_t = K*(1-u)*nabla^2(u) using PhiFlow.

    Uses explicit Euler for the full nonlinear equation.
    Small dt required for stability since the effective diffusion coefficient
    K*(1-u) varies spatially.

    Args:
        initial_field: 2D numpy array, shape (ny, nx)
        K: Diffusion coefficient
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep (keep small for stability)
        save_interval: Snapshot interval (None = every step)

    Returns:
        Tuple of:
        - field_history: List of 2D numpy arrays at saved timesteps
        - times: Array of time values
        - x: 1D array of x-coordinates
        - y: 1D array of y-coordinates
    """
    ny, nx = initial_field.shape
    Lx, Ly = domain_size

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    domain = Box(x=(0, Lx), y=(0, Ly))  # type: ignore[call-arg]

    u_field = math.tensor(initial_field, spatial("y,x"))
    u = CenteredGrid(u_field, extrapolation.PERIODIC, bounds=domain)

    field_history: List[np.ndarray] = []
    times: List[float] = []

    if save_interval is None:
        save_interval = dt

    n_steps = int(t_end / dt)
    save_every = max(1, int(save_interval / dt))

    # Save initial condition
    u_data = math.reshaped_native(u.values, ["y", "x"])
    field_history.append(np.array(u_data))
    times.append(0.0)

    print("Starting nonlinear heat equation simulation:")
    print(f"  Domain: {Lx} x {Ly}")
    print(f"  Resolution: {nx} x {ny}")
    print(f"  Coefficient: K = {K}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")
    print(f"  Total steps: {n_steps}, saving every {save_every} steps")

    # Strang splitting:
    #   Half-step reaction: u -> u (no change, but we record u for the correction)
    #   Full-step linear diffusion: u_t = K * nabla^2(u)
    #   Apply correction: scale the diffusion increment by (1 - u_before)
    #
    # More precisely:
    #   u_diffused = diffuse(u, K, dt)       # what linear diffusion would give
    #   delta_u = u_diffused - u             # the diffusion increment
    #   u_new = u + (1 - u) * delta_u        # scale by (1 - u)

    for step in range(1, n_steps + 1):
        t = step * dt

        # Store current state
        u_before = u

        # Full-step linear diffusion
        u_diffused = diffuse.implicit(u, K, dt, solve=Solve("CG", 1e-5, x0=None))

        # Compute diffusion increment and apply nonlinear correction
        delta_u = u_diffused.values - u_before.values
        correction = (1.0 - u_before.values) * delta_u
        u_new = u_before.values + correction

        u = CenteredGrid(u_new, extrapolation.PERIODIC, bounds=domain)

        if step % save_every == 0:
            u_data = math.reshaped_native(u.values, ["y", "x"])
            field_history.append(np.array(u_data))
            times.append(t)

            if step % (save_every * 10) == 0:
                u_mean = float(math.mean(u.values))
                print(f"  t = {t:.3f} / {t_end}  |  <u> = {u_mean:.6f}")

    print(f"Simulation complete. Saved {len(field_history)} snapshots.")

    return field_history, np.array(times), x, y


def solve_nl_heat_with_params(
    ic_params: dict[str, Any],
    simulation_params: NLHeatSimParams | dict[str, Any],
) -> dict[str, Any]:
    """
    High-level interface: generate IC from parameters and solve nonlinear heat equation.

    Args:
        ic_params: Dict with 'type' and type-specific parameters,
                   OR 'u_init' for custom IC
        simulation_params: NLHeatSimParams dataclass or dict

    Returns:
        Dict with field_history, times, x, y, ic_params, simulation_params
    """
    from src.data.initial_conditions_heat import create_heat_ic

    sim_dict: dict[str, Any]
    if hasattr(simulation_params, '__dataclass_fields__'):
        sim_dict = asdict(cast(NLHeatSimParams, simulation_params))
    else:
        sim_dict = cast(dict[str, Any], simulation_params)

    ny, nx = sim_dict["resolution"]
    Lx, Ly = sim_dict["domain_size"]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    generated_params: dict[str, Any]
    if ic_params.get("type") == "custom":
        u_init = ic_params["u_init"]
        generated_params = {}
    else:
        u_init, generated_params = create_heat_ic(ic_params, x, y)

    field_history, times, x_out, y_out = solve_nl_heat(
        initial_field=u_init,
        K=sim_dict["K"],
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
