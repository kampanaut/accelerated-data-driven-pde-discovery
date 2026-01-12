"""
Brusselator reaction-diffusion solver using PhiFlow.

The Brusselator model:
    ∂A/∂t = D_A ∇²A + k₁ - (k₂ + 1)A + A²B
    ∂B/∂t = D_B ∇²B + k₂A - A²B

where:
    A, B: concentrations
    D_A, D_B: diffusion coefficients
    k₁, k₂: reaction rate constants

Steady state: A = k₁, B = k₂/k₁
Turing instability requires D_B > D_A and specific parameter relationships.
"""

import numpy as np
from typing import Tuple, List, Dict
from phi.jax.flow import *


def brusselator_reaction(A: np.ndarray, B: np.ndarray, k1: float, k2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reaction terms for Brusselator.

    R_A = k₁ - (k₂ + 1)A + A²B
    R_B = k₂A - A²B

    Args:
        A, B: Concentration fields
        k1, k2: Reaction rate constants

    Returns:
        (R_A, R_B): Reaction terms for each species
    """
    A_squared_B = A * A * B
    R_A = k1 - (k2 + 1) * A + A_squared_B
    R_B = k2 * A - A_squared_B
    return R_A, R_B


def solve_brusselator(
    initial_concentration: Tuple[np.ndarray, np.ndarray],
    D_A: float,
    D_B: float,
    k1: float,
    k2: float,
    domain_size: Tuple[float, float],
    t_end: float,
    dt: float,
    save_interval: float = None
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D Brusselator reaction-diffusion equations using PhiFlow.

    Uses operator splitting: diffusion (implicit, unconditionally stable) then reaction (explicit Euler).

    Args:
        initial_concentration: Tuple of (A, B) numpy arrays, shape (ny, nx)
        D_A: Diffusion coefficient for species A
        D_B: Diffusion coefficient for species B
        k1: Reaction rate constant (feed rate)
        k2: Reaction rate constant
        domain_size: (Lx, Ly) physical domain size
        t_end: Final simulation time
        dt: Timestep for time integration
        save_interval: How often to save snapshots. If None, save every step.

    Returns:
        Tuple of:
        - concentration_history: List of (A, B) tuples at saved timesteps
        - times: Array of time values for saved snapshots
        - x: 1D array of x-coordinates
        - y: 1D array of y-coordinates
    """
    A_init, B_init = initial_concentration
    ny, nx = A_init.shape
    Lx, Ly = domain_size

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Create domain box
    domain = Box(x=(0, Lx), y=(0, Ly))

    # Initialize concentration fields as CenteredGrids
    A_field = math.tensor(A_init, spatial('y,x'))
    B_field = math.tensor(B_init, spatial('y,x'))

    A = CenteredGrid(A_field, extrapolation.PERIODIC, bounds=domain)
    B = CenteredGrid(B_field, extrapolation.PERIODIC, bounds=domain)

    # Initialize storage
    concentration_history = []
    times = []

    # Determine save frequency
    if save_interval is None:
        save_interval = dt

    n_steps = int(t_end / dt)
    save_every = max(1, int(save_interval / dt))

    # Save initial condition
    A_data = math.reshaped_native(A.values, ['y', 'x'])
    B_data = math.reshaped_native(B.values, ['y', 'x'])
    concentration_history.append((np.array(A_data), np.array(B_data)))
    times.append(0.0)

    print(f"Starting Brusselator simulation:")
    print(f"  Domain: {Lx} × {Ly}")
    print(f"  Resolution: {nx} × {ny}")
    print(f"  Diffusion: D_A = {D_A}, D_B = {D_B}")
    print(f"  Reaction: k₁ = {k1}, k₂ = {k2}")
    print(f"  Steady state: A* = {k1:.4f}, B* = {k2/k1:.4f}")
    print(f"  Time: [0, {t_end}] with dt = {dt}")
    print(f"  Total steps: {n_steps}, saving every {save_every} steps")

    # Time integration loop (operator splitting)
    for step in range(1, n_steps + 1):
        t = step * dt

        # Step 1: Diffusion (using PhiFlow's implicit diffusion - unconditionally stable)
        A = diffuse.implicit(A, D_A, dt, solve=Solve('CG', 1e-5, x0=None))
        B = diffuse.implicit(B, D_B, dt, solve=Solve('CG', 1e-5, x0=None))

        # Step 2: Reaction (explicit Euler on the reaction terms)
        A_np = np.array(math.reshaped_native(A.values, ['y', 'x']))
        B_np = np.array(math.reshaped_native(B.values, ['y', 'x']))

        R_A, R_B = brusselator_reaction(A_np, B_np, k1, k2)

        A_np = A_np + dt * R_A
        B_np = B_np + dt * R_B

        # Ensure non-negative concentrations (physical constraint)
        A_np = np.maximum(A_np, 0.0)
        B_np = np.maximum(B_np, 0.0)

        # Convert back to PhiFlow grids
        A = CenteredGrid(math.tensor(A_np, spatial('y,x')), extrapolation.PERIODIC, bounds=domain)
        B = CenteredGrid(math.tensor(B_np, spatial('y,x')), extrapolation.PERIODIC, bounds=domain)

        # Save snapshot at specified intervals
        if step % save_every == 0:
            A_data = math.reshaped_native(A.values, ['y', 'x'])
            B_data = math.reshaped_native(B.values, ['y', 'x'])
            concentration_history.append((np.array(A_data), np.array(B_data)))
            times.append(t)

            if step % (save_every * 10) == 0:  # Progress update
                A_mean = np.mean(A_np)
                B_mean = np.mean(B_np)
                print(f"  t = {t:.3f} / {t_end}  |  <A> = {A_mean:.4f}, <B> = {B_mean:.4f}")

    print(f"Simulation complete. Saved {len(concentration_history)} snapshots.")

    return concentration_history, np.array(times), x, y


def solve_brusselator_with_params(
    ic_params: dict,
    simulation_params: dict
) -> dict:
    """
    High-level interface: generate IC from parameters and solve Brusselator.

    Args:
        ic_params: Dict with keys:
            - 'type': IC type name
            - type-specific parameters
            - OR 'A_init', 'B_init' for custom IC
        simulation_params: Dict with keys:
            - 'D_A': Diffusion coefficient for A
            - 'D_B': Diffusion coefficient for B
            - 'k1': Reaction rate constant
            - 'k2': Reaction rate constant
            - 'domain_size': (Lx, Ly)
            - 'resolution': (ny, nx)
            - 't_end': Final time
            - 'dt': Timestep
            - 'save_interval': Snapshot interval

    Returns:
        Dict containing:
        - 'concentration_history': List of (A, B) tuples
        - 'times': Time values
        - 'x', 'y': Coordinate arrays
        - 'ic_params': Copy of IC parameters
        - 'simulation_params': Copy of simulation parameters
    """
    # Import Brusselator ICs (will be created next)
    from src.data.initial_conditions_brusselator import create_brusselator_ic

    # Extract parameters
    ny, nx = simulation_params['resolution']
    Lx, Ly = simulation_params['domain_size']

    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Generate initial condition
    if ic_params.get('type') == 'custom':
        A_init = ic_params['A_init']
        B_init = ic_params['B_init']
        generated_params = {}
    else:
        A_init, B_init, generated_params = create_brusselator_ic(ic_params, x, y)

    # Solve Brusselator
    concentration_history, times, x_out, y_out = solve_brusselator(
        initial_concentration=(A_init, B_init),
        D_A=simulation_params['D_A'],
        D_B=simulation_params['D_B'],
        k1=simulation_params['k1'],
        k2=simulation_params['k2'],
        domain_size=simulation_params['domain_size'],
        t_end=simulation_params['t_end'],
        dt=simulation_params['dt'],
        save_interval=simulation_params.get('save_interval')
    )

    return {
        'concentration_history': concentration_history,
        'times': times,
        'x': x_out,
        'y': y_out,
        'ic_params': ic_params.copy(),
        'simulation_params': simulation_params.copy(),
        'generated_params': generated_params
    }
