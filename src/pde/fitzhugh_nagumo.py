"""
FitzHugh-Nagumo reaction-diffusion solver using Dedalus spectral methods.

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

IMEX splitting:
    LHS (implicit): diffusion + linear damping of v (eps*a*v)
    RHS (explicit): nonlinear terms (u - u^3 - v, eps*(u - b))
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
class FHNSimParams:
    """Simulation parameters for FitzHugh-Nagumo solver."""

    D_u: float  # Diffusion coefficient for activator
    D_v: float  # Diffusion coefficient for recovery
    eps: float  # Timescale separation
    a: float  # Recovery coupling
    b: float  # Excitability threshold
    domain_size: Tuple[float, float]  # (Lx, Ly)
    resolution: Tuple[int, int]  # (ny, nx)
    t_end: float  # Final simulation time
    dt: float  # Timestep
    save_interval: Optional[float] = None


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
    task_name: str = "",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D FitzHugh-Nagumo reaction-diffusion equations using Dedalus.

    IMEX splitting: linear terms implicit (LHS), nonlinear terms explicit (RHS).

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

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    # --- Dedalus setup ---
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=nx, bounds=(0, Lx), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=ny, bounds=(0, Ly), dealias=3 / 2)

    u = dist.Field(name="u", bases=(xbasis, ybasis))
    v = dist.Field(name="v", bases=(xbasis, ybasis))
    u["g"] = u_init
    v["g"] = v_init

    # IMEX: LHS = implicit linear, RHS = explicit nonlinear
    #   u_t = D_u*lap(u) + u - u^3 - v
    #   v_t = D_v*lap(v) + eps*(u - a*v - b)
    #
    # Rearranged:
    #   dt(u) - D_u*lap(u)           = u - u*u*u - v
    #   dt(v) - D_v*lap(v) + eps*a*v = eps*(u - b)
    #
    # Note: eps*a*v is linear in v, so it goes on LHS for implicit stability.
    # u's linear term (+u) has positive sign (destabilizing), so we leave it
    # on the RHS — implicit treatment of a positive-definite term doesn't help.
    problem = d3.IVP(
        [u, v],
        namespace={"u": u, "v": v, "D_u": D_u, "D_v": D_v, "eps": eps, "a": a, "b": b},
    )
    problem.add_equation("dt(u) - D_u*lap(u) = u - u*u*u - v")
    problem.add_equation("dt(v) - D_v*lap(v) + eps*a*v = eps*(u - b)")
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
    field_history.append((np.array(u["g"]).copy(), np.array(v["g"]).copy()))
    times.append(0.0)

    tag = f"[{task_name}] " if task_name else ""

    print(f"{tag}Starting FitzHugh-Nagumo simulation:")
    print(f"{tag}  Domain: {Lx} x {Ly}")
    print(f"{tag}  Resolution: {nx} x {ny}")
    print(f"{tag}  Diffusion: D_u = {D_u}, D_v = {D_v}")
    print(f"{tag}  Kinetics: eps = {eps}, a = {a}, b = {b}")
    print(f"{tag}  Time: [0, {t_end}] with dt = {dt}")

    while solver.proceed:
        solver.step(dt)
        step += 1

        if step % save_every == 0:
            u.change_scales(1)
            v.change_scales(1)
            u_snap = np.array(u["g"]).copy()
            v_snap = np.array(v["g"]).copy()

            if not (np.isfinite(np.mean(u_snap)) and np.isfinite(np.mean(v_snap))):
                raise RuntimeError(
                    f"{tag}NaN/Inf detected at t={solver.sim_time:.3f}, aborting"
                )

            field_history.append((u_snap, v_snap))
            times.append(solver.sim_time)

            if step % (save_every * 10) == 0:
                print(
                    f"{tag}  t = {solver.sim_time:.3f} / {t_end}  |  <u> = {np.mean(u_snap):.4f}, <v> = {np.mean(v_snap):.4f}"
                )

    print(f"{tag}Simulation complete. Saved {len(field_history)} snapshots.")

    return field_history, np.array(times), x, y


def solve_fhn_with_params(
    ic_params: dict[str, Any],
    simulation_params: FHNSimParams | dict[str, Any],
    task_name: str = "",
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
    if hasattr(simulation_params, "__dataclass_fields__"):
        sim_dict = asdict(cast(FHNSimParams, simulation_params))
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
