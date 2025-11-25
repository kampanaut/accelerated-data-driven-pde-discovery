"""
Initial condition generators for fluid simulations.

This module provides functions to generate initial velocity fields from
vorticity specifications (e.g., Gaussian vortex bumps).
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple


def gaussian_vortex_ic(
    center: Tuple[float, float],
    width: float,
    strength: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate velocity field from a Gaussian vorticity distribution.

    The process:
    1. Create Gaussian vorticity: ω(r) = strength * exp(-r²/width²)
    2. Solve Poisson equation for stream function: ∇²ψ = -ω
    3. Compute velocity from stream function: u = -∂ψ/∂y, v = ∂ψ/∂x

    Args:
        center: (x_center, y_center) position of vortex
        width: Width parameter σ of the Gaussian
        strength: Vortex strength (circulation)
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates

    Returns:
        Tuple of (u, v) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
    """
    # Create 2D coordinate grid
    X, Y = np.meshgrid(x, y)

    # Compute distance from vortex center
    r_squared = (X - center[0])**2 + (Y - center[1])**2

    # Gaussian vorticity distribution
    vorticity = strength * np.exp(-r_squared / (2 * width**2))

    # Solve for stream function via Poisson equation
    psi = solve_poisson_2d(vorticity, x, y)

    # Compute velocity from stream function
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    u = -np.gradient(psi, dy, axis=0)  # u = -∂ψ/∂y
    v = np.gradient(psi, dx, axis=1)    # v = ∂ψ/∂x

    return u, v


def solve_poisson_2d(
    rhs: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    periodic: bool = True
) -> np.ndarray:
    """
    Solve 2D Poisson equation: ∇²φ = rhs

    Uses finite differences with periodic or Dirichlet boundary conditions.

    Args:
        rhs: Right-hand side of Poisson equation, shape (ny, nx)
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        periodic: If True, use periodic BCs. If False, use φ=0 on boundaries

    Returns:
        Solution φ, shape (ny, nx)
    """
    ny, nx = rhs.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if periodic:
        # For periodic BCs, use FFT-based solver (much faster)
        return solve_poisson_periodic(rhs, dx, dy)
    else:
        # For Dirichlet BCs, use sparse linear system
        return solve_poisson_dirichlet(rhs, dx, dy)


def solve_poisson_periodic(
    rhs: np.ndarray,
    dx: float,
    dy: float
) -> np.ndarray:
    """
    Solve 2D Poisson equation with periodic BCs using FFT.

    The Poisson equation ∇²φ = f becomes:
    -(kx² + ky²) φ̂ = f̂
    in Fourier space, where φ̂ and f̂ are Fourier transforms.

    Args:
        rhs: Right-hand side, shape (ny, nx)
        dx, dy: Grid spacings

    Returns:
        Solution, shape (ny, nx)
    """
    ny, nx = rhs.shape

    # Wavenumbers
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky)

    # Laplacian in Fourier space
    k_squared = KX**2 + KY**2

    # Avoid division by zero at k=0 (constant mode)
    k_squared[0, 0] = 1.0

    # Solve in Fourier space
    rhs_hat = np.fft.fft2(rhs)
    phi_hat = -rhs_hat / k_squared

    # Set constant mode to zero (fix gauge freedom)
    phi_hat[0, 0] = 0.0

    # Transform back to real space
    phi = np.fft.ifft2(phi_hat).real

    return phi


def solve_poisson_dirichlet(
    rhs: np.ndarray,
    dx: float,
    dy: float
) -> np.ndarray:
    """
    Solve 2D Poisson equation with Dirichlet BCs (φ=0 on boundary).

    Uses finite difference discretization and sparse solver.

    Args:
        rhs: Right-hand side, shape (ny, nx)
        dx, dy: Grid spacings

    Returns:
        Solution, shape (ny, nx)
    """
    ny, nx = rhs.shape

    # Interior points only (boundary is fixed at 0)
    n_interior = (ny - 2) * (nx - 2)

    # Build sparse matrix for 5-point stencil
    # φ_xx + φ_yy = (φ_{i+1,j} + φ_{i-1,j} - 2φ_{i,j})/dx² +
    #                (φ_{i,j+1} + φ_{i,j-1} - 2φ_{i,j})/dy²

    diag_main = -2.0 / dx**2 - 2.0 / dy**2
    diag_x = 1.0 / dx**2
    diag_y = 1.0 / dy**2

    diagonals = [
        np.full(n_interior, diag_main),      # main diagonal
        np.full(n_interior - 1, diag_x),      # super-diagonal (x-direction)
        np.full(n_interior - 1, diag_x),      # sub-diagonal (x-direction)
        np.full(n_interior - (nx-2), diag_y), # super-diagonal (y-direction)
        np.full(n_interior - (nx-2), diag_y), # sub-diagonal (y-direction)
    ]

    offsets = [0, 1, -1, nx-2, -(nx-2)]

    A = diags(diagonals, offsets, shape=(n_interior, n_interior), format='csr')

    # Flatten RHS (interior points only)
    b = rhs[1:-1, 1:-1].flatten()

    # Solve sparse system
    phi_interior = spsolve(A, b)

    # Reconstruct full solution with boundary conditions
    phi = np.zeros((ny, nx))
    phi[1:-1, 1:-1] = phi_interior.reshape((ny - 2, nx - 2))

    return phi


def multi_vortex_ic(
    vortex_params: list,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate velocity field from multiple Gaussian vortices.

    Superposition of vorticity fields from multiple vortices.

    Args:
        vortex_params: List of dicts, each with keys:
            - 'center': (x, y) position
            - 'width': Gaussian width
            - 'strength': Vortex circulation
        x, y: Coordinate arrays

    Returns:
        Tuple of (u, v) velocity components
    """
    # Initialize total vorticity as zero
    X, Y = np.meshgrid(x, y)
    vorticity_total = np.zeros_like(X)

    # Sum contributions from all vortices
    for params in vortex_params:
        center = params['center']
        width = params['width']
        strength = params['strength']

        r_squared = (X - center[0])**2 + (Y - center[1])**2
        vorticity_total += strength * np.exp(-r_squared / (2 * width**2))

    # Solve for stream function
    psi = solve_poisson_2d(vorticity_total, x, y, periodic=True)

    # Compute velocity
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    u = -np.gradient(psi, dy, axis=0)
    v = np.gradient(psi, dx, axis=1)

    return u, v


def taylor_green_vortex(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Taylor-Green vortex initial condition.

    Classic exact solution (at t=0) often used for testing N-S solvers:
    u(x, y, 0) = -sin(x) cos(y)
    v(x, y, 0) = cos(x) sin(y)

    Args:
        x, y: Coordinate arrays (assumed to be in [0, 2π])
        amplitude: Velocity amplitude

    Returns:
        Tuple of (u, v) velocity components
    """
    X, Y = np.meshgrid(x, y)

    u = -amplitude * np.sin(X) * np.cos(Y)
    v = amplitude * np.cos(X) * np.sin(Y)

    return u, v


def shear_layer_ic(
    y_center: float,
    thickness: float,
    velocity_jump: float,
    perturbation_amplitude: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate shear layer initial condition (Kelvin-Helmholtz setup).

    Creates a velocity jump across y = y_center with tanh profile.
    Optional sinusoidal perturbation triggers instability.

    Vorticity: ω = dU/dy where U(y) = (velocity_jump/2) * tanh((y - y_center) / thickness)

    Args:
        y_center: Location of shear layer
        thickness: Thickness of transition region
        velocity_jump: Velocity difference across layer
        perturbation_amplitude: Amplitude of sinusoidal perturbation
        x, y: Coordinate arrays

    Returns:
        Tuple of (u, v) velocity components
    """
    X, Y = np.meshgrid(x, y)

    # Base flow: u varies with y, v = 0
    u_base = (velocity_jump / 2) * np.tanh((Y - y_center) / thickness)

    # Add sinusoidal perturbation to trigger instability
    v_pert = perturbation_amplitude * np.sin(2 * np.pi * X / (x[-1] - x[0]))

    u = u_base
    v = v_pert

    return u, v


def lamb_oseen_vortex_ic(
    center: Tuple[float, float],
    core_radius: float,
    circulation: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lamb-Oseen vortex (smooth vortex with finite core).

    Velocity profile: v_θ(r) = (Γ / 2πr) * (1 - exp(-r² / a²))
    where Γ is circulation and a is core radius.

    This is a more realistic vortex model than Gaussian.

    Args:
        center: (x, y) position of vortex center
        core_radius: Core radius a
        circulation: Total circulation Γ
        x, y: Coordinate arrays

    Returns:
        Tuple of (u, v) velocity components
    """
    X, Y = np.meshgrid(x, y)

    # Distance from center
    dx_grid = X - center[0]
    dy_grid = Y - center[1]
    r = np.sqrt(dx_grid**2 + dy_grid**2)

    # Avoid division by zero at center
    r = np.maximum(r, 1e-10)

    # Tangential velocity magnitude
    v_theta = (circulation / (2 * np.pi * r)) * (1 - np.exp(-r**2 / core_radius**2))

    # Convert to Cartesian components: u = -v_θ * sin(θ), v = v_θ * cos(θ)
    # sin(θ) = dy/r, cos(θ) = dx/r
    u = -v_theta * (dy_grid / r)
    v = v_theta * (dx_grid / r)

    return u, v


def dipole_vortex_ic(
    center: Tuple[float, float],
    separation: float,
    width: float,
    strength: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dipole (vortex pair) - two counter-rotating vortices.

    Creates a pair of vortices with opposite circulation,
    separated horizontally. This creates a self-propelling structure.

    Args:
        center: (x, y) midpoint between the two vortices
        separation: Distance between vortex centers
        width: Gaussian width of each vortex
        strength: Circulation strength (one positive, one negative)
        x, y: Coordinate arrays

    Returns:
        Tuple of (u, v) velocity components
    """
    # Two vortices: one at center - separation/2, one at center + separation/2
    vortex_params = [
        {
            'center': (center[0] - separation / 2, center[1]),
            'width': width,
            'strength': strength
        },
        {
            'center': (center[0] + separation / 2, center[1]),
            'width': width,
            'strength': -strength  # Opposite sign
        }
    ]

    return multi_vortex_ic(vortex_params, x, y)


def perturbed_uniform_flow_ic(
    u_mean: float,
    v_mean: float,
    perturbation_amplitude: float,
    perturbation_wavelength: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uniform flow with small random perturbations.

    Useful for studying transition to turbulence.

    Args:
        u_mean: Mean x-velocity
        v_mean: Mean y-velocity
        perturbation_amplitude: RMS amplitude of perturbations
        perturbation_wavelength: Dominant wavelength of perturbations
        x, y: Coordinate arrays

    Returns:
        Tuple of (u, v) velocity components
    """
    X, Y = np.meshgrid(x, y)

    # Random Fourier modes
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility

    k_pert = 2 * np.pi / perturbation_wavelength

    # Generate random phases
    n_modes = 5
    u_pert = np.zeros_like(X)
    v_pert = np.zeros_like(Y)

    for i in range(n_modes):
        phase_u = rng.uniform(0, 2 * np.pi)
        phase_v = rng.uniform(0, 2 * np.pi)
        kx = k_pert * rng.uniform(0.5, 1.5)
        ky = k_pert * rng.uniform(0.5, 1.5)

        u_pert += np.sin(kx * X + ky * Y + phase_u)
        v_pert += np.sin(kx * X + ky * Y + phase_v)

    # Normalize perturbations
    u_pert = perturbation_amplitude * u_pert / np.sqrt(n_modes)
    v_pert = perturbation_amplitude * v_pert / np.sqrt(n_modes)

    u = u_mean + u_pert
    v = v_mean + v_pert

    return u, v


def random_vortex_soup_ic(
    n_vortices: int,
    strength_range: Tuple[float, float],
    width_range: Tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random collection of vortices (turbulent-like initial state).

    Places n_vortices at random locations with random strengths and widths.
    Useful for generating diverse training data.

    Args:
        n_vortices: Number of vortices to place
        strength_range: (min, max) circulation strength
        width_range: (min, max) vortex width
        x, y: Coordinate arrays
        seed: Random seed for reproducibility

    Returns:
        Tuple of (u, v) velocity components
    """
    rng = np.random.RandomState(seed)

    domain_x = (x[0], x[-1])
    domain_y = (y[0], y[-1])

    vortex_params = []

    for _ in range(n_vortices):
        center_x = rng.uniform(domain_x[0], domain_x[1])
        center_y = rng.uniform(domain_y[0], domain_y[1])
        strength = rng.uniform(strength_range[0], strength_range[1])
        width = rng.uniform(width_range[0], width_range[1])

        vortex_params.append({
            'center': (center_x, center_y),
            'width': width,
            'strength': strength
        })

    return multi_vortex_ic(vortex_params, x, y)


def von_karman_street_ic(
    n_vortices: int,
    spacing: float,
    offset: float,
    width: float,
    strength: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Von Kármán vortex street (alternating wake pattern).

    Creates two parallel rows of vortices with opposite circulation,
    staggered to create the classic wake instability pattern.

    Args:
        n_vortices: Number of vortices (will be split between two rows)
        spacing: Horizontal spacing between vortices in same row
        offset: Vertical distance between the two rows
        width: Gaussian width of each vortex
        strength: Circulation magnitude (alternates sign)
        x, y: Coordinate arrays

    Returns:
        Tuple of (u, v) velocity components
    """
    domain_x = x[-1] - x[0]
    domain_y = y[-1] - y[0]
    center_y = domain_y / 2

    vortex_params = []

    # Determine number of vortices per row
    n_per_row = n_vortices // 2

    for i in range(n_per_row):
        x_pos = x[0] + spacing * (i + 0.5)

        # Top row (positive circulation)
        vortex_params.append({
            'center': (x_pos, center_y + offset / 2),
            'width': width,
            'strength': strength
        })

        # Bottom row (negative circulation, staggered by spacing/2)
        vortex_params.append({
            'center': (x_pos + spacing / 2, center_y - offset / 2),
            'width': width,
            'strength': -strength
        })

    return multi_vortex_ic(vortex_params, x, y)
