"""
Heat equation solver using PhiFlow.

The heat equation:
    u_t = D * nabla^2(u)

where:
    u: scalar field
    D: diffusion coefficient

The simplest possible PDE. No reaction, no nonlinearity.
A Gaussian bump spreads out over time, smooth features decay exponentially.
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
class HeatSimParams:
    """Simulation parameters for heat equation solver."""

    D: float  # Diffusion coefficient
    domain_size: Tuple[float, float]  # (Lx, Ly)
    resolution: Tuple[int, int]  # (ny, nx)
    t_end: float  # Final simulation time
    dt: float  # Timestep
    save_interval: Optional[float] = None


def solve_heat(
    initial_field: np.ndarray,
    D: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: Optional[float] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D heat equation using PhiFlow implicit diffusion.

    No Strang splitting needed — there's no reaction term.
    Just implicit diffusion each timestep.

    Args:
        initial_field: 2D numpy array, shape (ny, nx)
        D: Diffusion coefficient
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep
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

    print("Starting heat equation simulation:")
    print(f"  Domain: {Lx} x {Ly}")
    print(f"  Resolution: {nx} x {ny}")
    print(f"  Diffusion: D = {D}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")
    print(f"  Total steps: {n_steps}, saving every {save_every} steps")

    for step in range(1, n_steps + 1):
        t = step * dt

        # Pure diffusion — no reaction step needed
        u = diffuse.implicit(u, D, dt, solve=Solve("CG", 1e-5, x0=None))

        if step % save_every == 0:
            u_data = math.reshaped_native(u.values, ["y", "x"])
            field_history.append(np.array(u_data))
            times.append(t)

            if step % (save_every * 10) == 0:
                u_mean = float(math.mean(u.values))
                print(f"  t = {t:.3f} / {t_end}  |  <u> = {u_mean:.6f}")

    print(f"Simulation complete. Saved {len(field_history)} snapshots.")

    return field_history, np.array(times), x, y


def solve_heat_with_params(
    ic_params: dict[str, Any],
    simulation_params: HeatSimParams | dict[str, Any],
) -> dict[str, Any]:
    """
    High-level interface: generate IC from parameters and solve heat equation.

    Args:
        ic_params: Dict with 'type' and type-specific parameters,
                   OR 'u_init' for custom IC
        simulation_params: HeatSimParams dataclass or dict

    Returns:
        Dict with field_history, times, x, y, ic_params, simulation_params
    """
    from src.data.initial_conditions_heat import create_heat_ic

    sim_dict: dict[str, Any]
    if hasattr(simulation_params, '__dataclass_fields__'):
        sim_dict = asdict(cast(HeatSimParams, simulation_params))
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

    field_history, times, x_out, y_out = solve_heat(
        initial_field=u_init,
        D=sim_dict["D"],
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
