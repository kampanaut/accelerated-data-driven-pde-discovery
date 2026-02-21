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
from typing import Tuple, List


def solve_navier_stokes(
    initial_velocity: Tuple[np.ndarray, np.ndarray],
    nu: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: float,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D incompressible Navier-Stokes equations using Dedalus.

    Solves:
        u_t + (u . grad)u = -grad(p) + nu * lap(u)
        div(u) = 0

    Args:
        initial_velocity: Tuple of (u, v) numpy arrays, shape (ny, nx)
        nu: Kinematic viscosity
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep for time integration
        save_interval: How often to save snapshots.

    Returns:
        Tuple of:
        - velocity_history: List of (u, v) tuples at saved timesteps
        - times: Array of time values for saved snapshots
        - x: 1D array of x-coordinates
        - y: 1D array of y-coordinates
    """
    u_init, v_init = initial_velocity
    ny, nx = u_init.shape
    Lx, Ly = domain_size

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)

    # --- Dedalus setup ---
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
    ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=3/2)

    # Vector velocity field + scalar pressure + pressure gauge
    u_vec = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
    p = dist.Field(name='p', bases=(xbasis, ybasis))
    tau_p = dist.Field(name='tau_p')

    # Set initial velocity: component 0 = x-velocity, component 1 = y-velocity
    u_vec['g'][0] = u_init
    u_vec['g'][1] = v_init

    # IMEX: LHS = implicit (viscosity + pressure), RHS = explicit (advection)
    #   dt(u) + grad(p) - nu*lap(u) = -(u . grad)(u)
    #   div(u) + tau_p = 0
    #   integ(p) = 0
    problem = d3.IVP([u_vec, p, tau_p], namespace={'u': u_vec, 'p': p, 'tau_p': tau_p, 'nu': nu})
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0")
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = t_end

    # --- Snapshot loop ---
    velocity_history: List[Tuple[np.ndarray, np.ndarray]] = []
    times: List[float] = []

    save_every = max(1, int(save_interval / dt))
    step = 0

    # Save initial condition
    u_vec.change_scales(1)
    velocity_history.append(
        (np.array(u_vec['g'][0]).copy(), np.array(u_vec['g'][1]).copy())
    )
    times.append(0.0)

    print("Starting Navier-Stokes simulation:")
    print(f"  Domain: {Lx} x {Ly}")
    print(f"  Resolution: {nx} x {ny}")
    print(f"  Viscosity: nu = {nu}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")

    while solver.proceed:
        solver.step(dt)
        step += 1

        if step % save_every == 0:
            u_vec.change_scales(1)
            velocity_history.append(
                (np.array(u_vec['g'][0]).copy(), np.array(u_vec['g'][1]).copy())
            )
            times.append(solver.sim_time)

            if step % (save_every * 5) == 0:
                print(f"  t = {solver.sim_time:.3f} / {t_end}")

    print(f"Simulation complete. Saved {len(velocity_history)} snapshots.")

    return velocity_history, np.array(times), x, y


def solve_navier_stokes_with_params(ic_params: dict, simulation_params: dict) -> dict:
    """
    High-level interface: generate IC from parameters and solve N-S.

    Args:
        ic_params: Dict with keys:
            - 'type': 'gaussian_vortex', 'multi_vortex', or 'taylor_green'
            - type-specific parameters
        simulation_params: Dict with keys:
            - 'nu': viscosity
            - 'domain_size': (Lx, Ly)
            - 'resolution': (nx, ny)
            - 't_end': final time
            - 'dt': timestep
            - 'save_interval': snapshot interval

    Returns:
        Dict containing:
        - 'velocity_history': List of (u, v) tuples
        - 'times': Time values
        - 'x', 'y': Coordinate arrays
        - 'ic_params': Copy of IC parameters
        - 'simulation_params': Copy of simulation parameters
    """

    # Generate initial condition
    ic_type = ic_params["type"]

    if ic_type == "custom":
        u_init = ic_params["u_init"]
        v_init = ic_params["v_init"]
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    # Solve Navier-Stokes
    velocity_history, times, x_out, y_out = solve_navier_stokes(
        initial_velocity=(u_init, v_init),
        nu=simulation_params["nu"],
        domain_size=simulation_params["domain_size"],
        t_end=simulation_params["t_end"],
        dt=simulation_params["dt"],
        save_interval=simulation_params["save_interval"],
    )

    return {
        "velocity_history": velocity_history,
        "times": times,
        "x": x_out,
        "y": y_out,
        "ic_params": ic_params.copy(),
        "simulation_params": simulation_params.copy(),
    }
