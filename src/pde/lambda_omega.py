"""
Lambda-Omega reaction-diffusion solver using Dedalus spectral methods.

The Lambda-Omega model:
    u_t = D_u * nabla^2(u) + a*u - (u + c*v)*(u^2 + v^2)
    v_t = D_v * nabla^2(v) + a*v + (c*u - v)*(u^2 + v^2)

where:
    u, v: oscillatory fields
    D_u, D_v: diffusion coefficients
    a: linear growth rate (controls limit-cycle amplitude R_0 = sqrt(a))
    c: rotation/frequency parameter (controls spiral twist)

This is the normal form for oscillatory media near a Hopf bifurcation.
The origin (0,0) is unstable; any perturbation self-organizes into spiral waves.
Amplitude saturates at R_0 = sqrt(a), frequency omega = c*a - 1.

IMEX splitting:
    LHS (implicit): diffusion + linear growth (a*u, a*v)
    RHS (explicit): cubic nonlinearity
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import dedalus.public as d3
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Any, cast


# =============================================================================
# Dataclasses for type-safe configuration
# =============================================================================


@dataclass
class LOSimParams:
    """Simulation parameters for Lambda-Omega solver."""

    D_u: float  # Diffusion coefficient for u
    D_v: float  # Diffusion coefficient for v
    a: float    # Linear growth rate
    c: float    # Rotation/frequency parameter
    domain_size: Tuple[float, float]  # (Lx, Ly)
    resolution: Tuple[int, int]  # (ny, nx)
    t_end: float  # Final simulation time
    dt: float  # Timestep
    save_interval: Optional[float] = None


# =============================================================================
# Solver
# =============================================================================


def solve_lo(
    initial_fields: Tuple[np.ndarray, np.ndarray],
    D_u: float,
    D_v: float,
    a: float,
    c: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: Optional[float] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D Lambda-Omega reaction-diffusion equations using Dedalus.

    IMEX splitting: linear terms implicit (LHS), nonlinear terms explicit (RHS).

    Args:
        initial_fields: Tuple of (u, v) numpy arrays, shape (ny, nx)
        D_u: Diffusion coefficient for u
        D_v: Diffusion coefficient for v
        a: Linear growth rate
        c: Rotation parameter
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
    #   u_t = D_u*lap(u) + a*u - (u + c*v)*(u^2 + v^2)
    #   v_t = D_v*lap(v) + a*v + (c*u - v)*(u^2 + v^2)
    #
    # Rearranged:
    #   dt(u) - D_u*lap(u) - a*u = -(u + c*v)*(u*u + v*v)
    #   dt(v) - D_v*lap(v) - a*v =  (c*u - v)*(u*u + v*v)
    #
    # Note: -a*u on LHS means the linear growth is treated implicitly.
    # a > 0 makes this a destabilizing term, but implicit treatment
    # still helps stability by capturing it exactly.
    problem = d3.IVP([u, v], namespace={'u': u, 'v': v, 'D_u': D_u, 'D_v': D_v, 'a': a, 'c': c})
    problem.add_equation("dt(u) - D_u*lap(u) - a*u = -(u + c*v)*(u*u + v*v)")
    problem.add_equation("dt(v) - D_v*lap(v) - a*v =  (c*u - v)*(u*u + v*v)")
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = t_end

    # --- Snapshot loop ---
    field_history: List[Tuple[np.ndarray, np.ndarray]] = []
    times: List[float] = []

    if save_interval is None:
        save_interval = dt

    save_every = max(1, int(save_interval / dt))
    step = 0

    # Save initial condition
    u.change_scales(1)
    v.change_scales(1)
    field_history.append((np.array(u['g']).copy(), np.array(v['g']).copy()))
    times.append(0.0)

    print("Starting Lambda-Omega simulation:")
    print(f"  Domain: {Lx} x {Ly}")
    print(f"  Resolution: {nx} x {ny}")
    print(f"  Diffusion: D_u = {D_u}, D_v = {D_v}")
    print(f"  Kinetics: a = {a}, c = {c}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")

    while solver.proceed:
        solver.step(dt)
        step += 1

        if step % save_every == 0:
            u.change_scales(1)
            v.change_scales(1)
            field_history.append(
                (np.array(u['g']).copy(), np.array(v['g']).copy())
            )
            times.append(solver.sim_time)

            if step % (save_every * 10) == 0:
                u_mean = np.mean(u['g'])
                v_mean = np.mean(v['g'])
                print(
                    f"  t = {solver.sim_time:.3f} / {t_end}  |  <u> = {u_mean:.4f}, <v> = {v_mean:.4f}"
                )

    print(f"Simulation complete. Saved {len(field_history)} snapshots.")

    return field_history, np.array(times), x, y


def solve_lo_with_params(
    ic_params: dict[str, Any],
    simulation_params: LOSimParams | dict[str, Any],
) -> dict[str, Any]:
    """
    High-level interface: generate IC from parameters and solve Lambda-Omega.

    Args:
        ic_params: Dict with 'type' and type-specific parameters,
                   OR 'u_init', 'v_init' for custom IC
        simulation_params: LOSimParams dataclass or dict

    Returns:
        Dict with field_history, times, x, y, ic_params, simulation_params
    """
    from src.data.initial_conditions_lo import create_lo_ic

    sim_dict: dict[str, Any]
    if hasattr(simulation_params, '__dataclass_fields__'):
        sim_dict = asdict(cast(LOSimParams, simulation_params))
    else:
        sim_dict = cast(dict[str, Any], simulation_params)

    ny, nx = sim_dict["resolution"]
    Lx, Ly = sim_dict["domain_size"]

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    generated_params: dict[str, Any]
    if ic_params.get("type") == "custom":
        u_init = ic_params["u_init"]
        v_init = ic_params["v_init"]
        generated_params = {}
    else:
        u_init, v_init, generated_params = create_lo_ic(ic_params, x, y)

    field_history, times, x_out, y_out = solve_lo(
        initial_fields=(u_init, v_init),
        D_u=sim_dict["D_u"],
        D_v=sim_dict["D_v"],
        a=sim_dict["a"],
        c=sim_dict["c"],
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
