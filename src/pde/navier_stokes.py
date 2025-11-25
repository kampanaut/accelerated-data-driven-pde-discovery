"""
Navier-Stokes solver using PhiFlow.

This module wraps PhiFlow's incompressible fluid simulation to solve
2D Navier-Stokes equations with custom initial conditions.
"""

import numpy as np
from typing import Tuple, List
from phi.flow import *


def solve_navier_stokes(
    initial_velocity: Tuple[np.ndarray, np.ndarray],
    nu: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: float = None
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D incompressible Navier-Stokes equations using PhiFlow.

    Solves:
        u_t + (u·∇)u = -∇p + ν∇²u
        ∇·u = 0

    Args:
        initial_velocity: Tuple of (u, v) numpy arrays, shape (ny, nx)
        nu: Kinematic viscosity
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep for time integration
        save_interval: How often to save snapshots. If None, save every step.

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

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Convert initial velocity to PhiFlow format
    # PhiFlow uses staggered grids where u and v are offset
    # Create velocity as StaggeredGrid (proper way for vector fields in PhiFlow)

    # Create a domain box
    domain = Box(x=(0, Lx), y=(0, Ly))

    # Create initial velocity as StaggeredGrid from numpy arrays
    # StaggeredGrid expects a callable or specific format
    # Easier approach: use CenteredGrid with tensor wrapper
    velocity_field = math.tensor(np.stack([u_init, v_init], axis=-1), spatial('y,x'), channel('vector'))

    velocity = CenteredGrid(
        velocity_field,
        extrapolation.PERIODIC,
        bounds=domain
    )

    # Initialize storage
    velocity_history = []
    times = []

    # Determine save frequency
    if save_interval is None:
        save_interval = dt

    n_steps = int(t_end / dt)
    save_every = max(1, int(save_interval / dt))

    # Save initial condition
    vel_data = math.reshaped_native(velocity.values, ['y', 'x', 'vector'])
    velocity_history.append((vel_data[..., 0], vel_data[..., 1]))
    times.append(0.0)

    print(f"Starting Navier-Stokes simulation:")
    print(f"  Domain: {Lx} × {Ly}")
    print(f"  Resolution: {nx} × {ny}")
    print(f"  Viscosity: ν = {nu}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")
    print(f"  Total steps: {n_steps}, saving every {save_every} steps")

    # Time integration loop
    for step in range(1, n_steps + 1):
        t = step * dt

        # PhiFlow's incompressible Navier-Stokes step:
        # 1. Advection: move velocity along itself
        # 2. Diffusion: apply viscous diffusion
        # 3. Pressure projection: enforce incompressibility

        # Advection (semi-Lagrangian)
        velocity = advect.semi_lagrangian(velocity, velocity, dt)

        # Diffusion
        velocity = diffuse.explicit(velocity, nu, dt)

        # Pressure projection (make incompressible)
        velocity, pressure = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, x0=None))

        # Save snapshot at specified intervals
        if step % save_every == 0:
            vel_data = math.reshaped_native(velocity.values, ['y', 'x', 'vector'])
            velocity_history.append((vel_data[..., 0], vel_data[..., 1]))
            times.append(t)

            if step % (save_every * 5) == 0:  # Progress update
                print(f"  t = {t:.3f} / {t_end}")

    print(f"Simulation complete. Saved {len(velocity_history)} snapshots.")

    return velocity_history, np.array(times), x, y


def solve_navier_stokes_with_params(
    ic_params: dict,
    simulation_params: dict
) -> dict:
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
    from src.data.initial_conditions import gaussian_vortex_ic, multi_vortex_ic, taylor_green_vortex

    # Extract parameters
    nx, ny = simulation_params['resolution']
    Lx, Ly = simulation_params['domain_size']

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Generate initial condition
    ic_type = ic_params['type']

    if ic_type == 'gaussian_vortex':
        u_init, v_init = gaussian_vortex_ic(
            center=ic_params['center'],
            width=ic_params['width'],
            strength=ic_params['strength'],
            x=x, y=y
        )
    elif ic_type == 'multi_vortex':
        u_init, v_init = multi_vortex_ic(
            vortex_params=ic_params['vortices'],
            x=x, y=y
        )
    elif ic_type == 'taylor_green':
        u_init, v_init = taylor_green_vortex(
            x=x, y=y,
            amplitude=ic_params.get('amplitude', 1.0)
        )
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    # Solve Navier-Stokes
    velocity_history, times, x_out, y_out = solve_navier_stokes(
        initial_velocity=(u_init, v_init),
        nu=simulation_params['nu'],
        domain_size=simulation_params['domain_size'],
        t_end=simulation_params['t_end'],
        dt=simulation_params['dt'],
        save_interval=simulation_params.get('save_interval')
    )

    return {
        'velocity_history': velocity_history,
        'times': times,
        'x': x_out,
        'y': y_out,
        'ic_params': ic_params.copy(),
        'simulation_params': simulation_params.copy()
    }
