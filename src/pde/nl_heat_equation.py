"""
Nonlinear heat equation solver using Dedalus spectral methods.

The nonlinear heat equation:
    u_t = K * (1 - u) * nabla^2(u)

Rewritten as:
    u_t = K * nabla^2(u) - K * u * nabla^2(u)

where:
    u: scalar field
    K: diffusion coefficient

The (1 - u) factor makes diffusion state-dependent:
    - Where u is small (u << 1), diffusion is nearly K (like regular heat equation)
    - Where u approaches 1, diffusion shuts off
    - Where u > 1, diffusion reverses sign (anti-diffusion, can cause instabilities)

IMEX splitting:
    LHS (implicit): K * nabla^2(u) — linear diffusion
    RHS (explicit): -K * u * nabla^2(u) — nonlinear correction
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import dedalus.public as d3
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Any, cast


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
    task_name: str = "",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D nonlinear heat equation u_t = K*(1-u)*nabla^2(u) using Dedalus.

    IMEX: linear diffusion implicit (LHS), nonlinear correction explicit (RHS).

    Args:
        initial_field: 2D numpy array, shape (ny, nx)
        K: Diffusion coefficient
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
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=nx, bounds=(0, Lx), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=ny, bounds=(0, Ly), dealias=3 / 2)

    u = dist.Field(name="u", bases=(xbasis, ybasis))
    u["g"] = initial_field.T  # (ny, nx) → (nx, ny) for Dedalus axis convention

    # IMEX: LHS = implicit linear, RHS = explicit nonlinear
    #   u_t = K*lap(u) - K*u*lap(u)
    #
    # Rearranged:
    #   dt(u) - K*lap(u) = -K*u*lap(u)
    problem = d3.IVP([u], namespace={"u": u, "K": K})
    problem.add_equation("dt(u) - K*lap(u) = -K*u*lap(u)")
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
    field_history.append(np.array(u["g"]).T.copy())  # (nx, ny) → (ny, nx) for caller
    times.append(0.0)

    tag = f"[{task_name}] " if task_name else ""
    print(f"{tag}Starting nonlinear heat equation simulation:")
    print(f"  {tag}Domain: {Lx} x {Ly}")
    print(f"  {tag}Resolution: {nx} x {ny}")
    print(f"  {tag}Coefficient: K = {K}")
    print(f"  {tag}Time: [0, {t_end}] with dt = {dt}")

    while solver.proceed:
        solver.step(dt)
        step += 1

        if step % save_every == 0:
            u.change_scales(1)
            snapshot = np.array(u["g"]).T.copy()  # (nx, ny) → (ny, nx)

            if not np.isfinite(np.mean(snapshot)):
                raise RuntimeError(
                    f"{tag}NaN/Inf detected at t={solver.sim_time:.3f}, aborting"
                )

            field_history.append(snapshot)
            times.append(solver.sim_time)

            if step % (save_every * 10) == 0:
                print(
                    f"  {tag}t = {solver.sim_time:.3f} / {t_end}  |  <u> = {np.mean(snapshot):.6f}"
                )

    print(f"{tag}Simulation complete. Saved {len(field_history)} snapshots.")

    return field_history, np.array(times), x, y


def solve_nl_heat_with_params(
    ic_params: dict[str, Any],
    simulation_params: NLHeatSimParams | dict[str, Any],
    task_name: str = "",
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
    if hasattr(simulation_params, "__dataclass_fields__"):
        sim_dict = asdict(cast(NLHeatSimParams, simulation_params))
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

    field_history, times, x_out, y_out = solve_nl_heat(
        initial_field=u_init,
        K=sim_dict["K"],
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
