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
from typing import Tuple, List, Any


def solve_fhn(
    initial_fields: Tuple[np.ndarray, np.ndarray],
    simulation_params: dict[str, Any],
    task_name: str = "",
) -> dict[str, Any]:
    """
    Solve 2D FitzHugh-Nagumo reaction-diffusion equations using Dedalus.

    IMEX splitting: linear terms implicit (LHS), nonlinear terms explicit (RHS).

    Args:
        initial_fields: Tuple of (u, v) numpy arrays, shape (ny, nx)
        simulation_params: Dict with keys: D_u, D_v, eps, a, b, domain_size,
            resolution, t_end, dt, save_interval
        task_name: Optional label for log messages

    Returns:
        Dict containing:
        - 'field_history': List of (u, v) tuples
        - 'times': Time values array
        - 'x', 'y': Coordinate arrays
    """
    u_init, v_init = initial_fields
    ny, nx = u_init.shape

    D_u = simulation_params["D_u"]
    D_v = simulation_params["D_v"]
    eps = simulation_params["eps"]
    a = simulation_params["a"]
    b = simulation_params["b"]
    Lx, Ly = simulation_params["domain_size"]
    t_end = simulation_params["t_end"]
    dt = simulation_params["dt"]
    save_interval = simulation_params.get("save_interval")

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    # --- Dedalus setup ---
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=nx, bounds=(0, Lx), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=ny, bounds=(0, Ly), dealias=3 / 2)

    u = dist.Field(name="u", bases=(xbasis, ybasis))
    v = dist.Field(name="v", bases=(xbasis, ybasis))
    u["g"] = u_init.T  # (ny, nx) → (nx, ny) for Dedalus axis convention
    v["g"] = v_init.T

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
    field_history.append((np.array(u["g"]).T.copy(), np.array(v["g"]).T.copy()))  # (nx, ny) → (ny, nx)
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
            u_snap = np.array(u["g"]).T.copy()  # (nx, ny) → (ny, nx)
            v_snap = np.array(v["g"]).T.copy()

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

    return {
        "field_history": field_history,
        "times": np.array(times),
        "x": x,
        "y": y,
    }
