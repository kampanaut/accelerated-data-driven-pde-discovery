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
from typing import Tuple, List, Any


def solve_lo(
    initial_fields: Tuple[np.ndarray, np.ndarray],
    simulation_params: dict[str, Any],
    task_name: str = "",
) -> dict[str, Any]:
    """
    Solve 2D Lambda-Omega reaction-diffusion equations using Dedalus.

    IMEX splitting: linear terms implicit (LHS), nonlinear terms explicit (RHS).

    Args:
        initial_fields: Tuple of (u, v) numpy arrays, shape (ny, nx)
        simulation_params: Dict with keys: D_u, D_v, a, c, domain_size,
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
    a = simulation_params["a"]
    c = simulation_params["c"]
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
    problem = d3.IVP(
        [u, v], namespace={"u": u, "v": v, "D_u": D_u, "D_v": D_v, "a": a, "c": c}
    )
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
    field_history.append(
        (np.array(u["g"]).T.copy(), np.array(v["g"]).T.copy())
    )  # (nx, ny) → (ny, nx)
    times.append(0.0)

    tag = f"[{task_name}] " if task_name else ""

    print(f"{tag}Starting Lambda-Omega simulation:")
    print(f"{tag}  Domain: {Lx} x {Ly}")
    print(f"{tag}  Resolution: {nx} x {ny}")
    print(f"{tag}  Diffusion: D_u = {D_u}, D_v = {D_v}")
    print(f"{tag}  Kinetics: a = {a}, c = {c}")
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
