"""
Navier-Stokes solver using Dedalus spectral methods.

2D incompressible Navier-Stokes:
    u_t + (u . grad)u = -grad(p) + nu * lap(u)
    div(u) = 0

where:
    u: velocity vector field (u, v components)
    p: pressure (enforces incompressibility)
    nu: kinematic viscosity

Dedalus handles pressure projection through the equation system —
no separate projection step needed. The tau_p gauge variable
pins the mean pressure to zero.

IMEX splitting:
    LHS (implicit): viscous diffusion + pressure gradient
    RHS (explicit): nonlinear advection u . grad(u)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import dedalus.public as d3
from typing import Tuple, List, Any


def solve_ns(
    initial_fields: Tuple[np.ndarray, np.ndarray],
    simulation_params: dict[str, Any],
    task_name: str = "",
) -> dict[str, Any]:
    """
    Solve 2D incompressible Navier-Stokes equations using Dedalus.

    Solves:
        u_t + (u . grad)u = -grad(p) + nu * lap(u)
        div(u) = 0

    Extracts pressure at each snapshot for analytical target computation
    downstream (no more temporal central differences).

    Args:
        initial_fields: Tuple of (u, v) numpy arrays, shape (ny, nx)
        simulation_params: Dict with keys: nu, domain_size, resolution,
            t_end, dt, save_interval
        task_name: Optional label for log messages

    Returns:
        Dict containing:
        - 'velocity_history': List of (u, v) tuples
        - 'pressure_history': List of pressure arrays
        - 'times': Time values array
        - 'x', 'y': Coordinate arrays
    """
    u_init, v_init = initial_fields
    ny, nx = u_init.shape

    nu = simulation_params["nu"]
    Lx, Ly = simulation_params["domain_size"]
    t_end = simulation_params["t_end"]
    dt = simulation_params["dt"]
    save_interval = simulation_params["save_interval"]

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    # --- Dedalus setup ---
    coords = d3.CartesianCoordinates("x", "y")
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords["x"], size=nx, bounds=(0, Lx), dealias=3 / 2)
    ybasis = d3.RealFourier(coords["y"], size=ny, bounds=(0, Ly), dealias=3 / 2)

    # Vector velocity field + scalar pressure + pressure gauge
    u_vec = dist.VectorField(coords, name="u", bases=(xbasis, ybasis))
    p = dist.Field(name="p", bases=(xbasis, ybasis))
    tau_p = dist.Field(name="tau_p")

    # Set initial velocity: component 0 = x-velocity, component 1 = y-velocity
    u_vec["g"][0] = u_init.T  # (ny, nx) → (nx, ny) for Dedalus axis convention
    u_vec["g"][1] = v_init.T

    # IMEX: LHS = implicit (viscosity + pressure), RHS = explicit (advection)
    #   dt(u) + grad(p) - nu*lap(u) = -(u . grad)(u)
    #   div(u) + tau_p = 0
    #   integ(p) = 0
    problem = d3.IVP(
        [u_vec, p, tau_p], namespace={"u": u_vec, "p": p, "tau_p": tau_p, "nu": nu}
    )
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0")
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = t_end

    # --- Snapshot loop ---
    velocity_history: List[Tuple[np.ndarray, np.ndarray]] = []
    pressure_history: List[np.ndarray] = []
    times: List[float] = []

    save_every = max(1, int(save_interval / dt))
    step = 0

    # Save initial condition
    # Note: at t=0 before any timestep, pressure hasn't been solved yet.
    # We take one implicit step first, then save. But for consistency with
    # other solvers that save t=0, we save the IC velocity with p=0.
    # The first real pressure snapshot comes after the first save_every steps.
    u_vec.change_scales(1)
    p.change_scales(1)
    velocity_history.append(
        (np.array(u_vec["g"][0]).T.copy(), np.array(u_vec["g"][1]).T.copy())  # (nx, ny) → (ny, nx)
    )
    pressure_history.append(np.array(p["g"]).T.copy())
    times.append(0.0)

    tag = f"[{task_name}] " if task_name else ""

    print(f"{tag}Starting Navier-Stokes simulation:")
    print(f"{tag}  Domain: {Lx} x {Ly}")
    print(f"{tag}  Resolution: {nx} x {ny}")
    print(f"{tag}  Viscosity: nu = {nu}")
    print(f"{tag}  Time: [0, {t_end}] with dt = {dt}")

    while solver.proceed:
        solver.step(dt)
        step += 1

        if step % save_every == 0:
            u_vec.change_scales(1)
            p.change_scales(1)
            u_snap = np.array(u_vec["g"][0]).T.copy()  # (nx, ny) → (ny, nx)
            v_snap = np.array(u_vec["g"][1]).T.copy()
            p_snap = np.array(p["g"]).T.copy()

            if not (np.isfinite(np.mean(u_snap)) and np.isfinite(np.mean(v_snap))):
                raise RuntimeError(
                    f"{tag}NaN/Inf detected at t={solver.sim_time:.3f}, aborting"
                )

            velocity_history.append((u_snap, v_snap))
            pressure_history.append(p_snap)
            times.append(solver.sim_time)

            if step % (save_every * 5) == 0:
                print(f"{tag}  t = {solver.sim_time:.3f} / {t_end}")

    print(f"{tag}Simulation complete. Saved {len(velocity_history)} snapshots.")

    return {
        "velocity_history": velocity_history,
        "pressure_history": pressure_history,
        "times": np.array(times),
        "x": x,
        "y": y,
    }
