"""
Heat equation solver using Dedalus spectral methods.

The heat equation:
    u_t = D * nabla^2(u)

where:
    u: scalar field
    D: diffusion coefficient

The simplest possible PDE. No reaction, no nonlinearity.
A Gaussian bump spreads out over time, smooth features decay exponentially.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import dedalus.public as d3
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Any, cast


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
    task_name: str = "",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D heat equation using Dedalus spectral methods.

    Fully implicit (linear PDE) — unconditionally stable for any dt.

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

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    # --- Dedalus setup ---
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
    ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=3/2)

    u = dist.Field(name='u', bases=(xbasis, ybasis))
    u['g'] = initial_field

    problem = d3.IVP([u], namespace={'u': u, 'D': D})
    problem.add_equation("dt(u) - D*lap(u) = 0")
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = t_end

    # --- Snapshot loop ---
    field_history: List[np.ndarray] = []
    times: List[float] = []

    if save_interval is None:
        save_interval = dt

    save_every = max(1, int(save_interval / dt))
    step = 0

    # Save initial condition
    u.change_scales(1)
    field_history.append(np.array(u['g']).copy())
    times.append(0.0)

    tag = f"[{task_name}] " if task_name else ""
    print(f"{tag}Starting heat equation simulation:")
    print(f"  {tag}Domain: {Lx} x {Ly}")
    print(f"  {tag}Resolution: {nx} x {ny}")
    print(f"  {tag}Diffusion: D = {D}")
    print(f"  {tag}Time: [0, {t_end}] with dt = {dt}")

    while solver.proceed:
        solver.step(dt)
        step += 1

        if step % save_every == 0:
            u.change_scales(1)
            snapshot = np.array(u['g']).copy()

            if not np.isfinite(np.mean(snapshot)):
                raise RuntimeError(f"{tag}NaN/Inf detected at t={solver.sim_time:.3f}, aborting")

            field_history.append(snapshot)
            times.append(solver.sim_time)

            if step % (save_every * 10) == 0:
                print(f"  {tag}t = {solver.sim_time:.3f} / {t_end}  |  <u> = {np.mean(snapshot):.6f}")

    print(f"{tag}Simulation complete. Saved {len(field_history)} snapshots.")

    return field_history, np.array(times), x, y


def solve_heat_with_params(
    ic_params: dict[str, Any],
    simulation_params: HeatSimParams | dict[str, Any],
    task_name: str = "",
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

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

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
        task_name=task_name,
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
