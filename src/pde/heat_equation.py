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
from typing import Tuple, List, Any


def solve_heat(
    initial_fields: Tuple[np.ndarray],
    simulation_params: dict[str, Any],
    task_name: str = "",
) -> dict[str, Any]:
    """
    Solve 2D heat equation using Dedalus spectral methods.

    Fully implicit (linear PDE) — unconditionally stable for any dt.

    Args:
        initial_fields: 1-tuple of (u,) numpy array, shape (ny, nx)
        simulation_params: Dict with keys: D, domain_size, resolution,
            t_end, dt, save_interval
        task_name: Optional label for log messages

    Returns:
        Dict containing:
        - 'field_history': List of 2D numpy arrays
        - 'times': Time values array
        - 'x', 'y': Coordinate arrays
    """
    (initial_field,) = initial_fields
    ny, nx = initial_field.shape

    D = simulation_params["D"]
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
    u["g"] = initial_field.T  # (ny, nx) → (nx, ny) for Dedalus axis convention

    problem = d3.IVP([u], namespace={"u": u, "D": D})
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
    field_history.append(np.array(u["g"]).T.copy())  # (nx, ny) → (ny, nx) for caller
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

    return {
        "field_history": field_history,
        "times": np.array(times),
        "x": x,
        "y": y,
    }
