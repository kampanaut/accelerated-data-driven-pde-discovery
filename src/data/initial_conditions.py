"""
Initial condition generators for fluid simulations.

This module provides functions to generate initial velocity fields from
vorticity specifications (e.g., Gaussian vortex bumps).
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple


def gaussian_hill_ic(
    center: Tuple[float, float],
    width: float,
    strength: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate velocity field from a single Gaussian vorticity "hill".

    Creates a single Gaussian bump of vorticity, which produces a localized
    swirling flow pattern. Named "hill" because the vorticity profile looks
    like a smooth hill/bump in 2D.

    The process:
    1. Create Gaussian vorticity: ω(r) = strength * exp(-r²/width²)
    2. Solve Poisson equation for stream function: ∇²ψ = -ω
    3. Compute velocity from stream function: u = -∂ψ/∂y, v = ∂ψ/∂x

    Args:
        center: (x_center, y_center) position of the hill
        width: Width parameter σ of the Gaussian
        strength: Peak vorticity strength
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates

    Returns:
        Tuple of (u, v, params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - params: Dict with center, width, strength
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

    params = {
        'center': center,
        'width': width,
        'strength': strength
    }

    return u, v, params


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
) -> Tuple[np.ndarray, np.ndarray, list]:
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
        Tuple of (u, v, vortex_params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - vortex_params: List of dicts with actual parameters used
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

    return u, v, vortex_params


def taylor_green_vortex(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate Taylor-Green vortex initial condition.

    Classic exact solution (at t=0) often used for testing N-S solvers:
    u(x, y, 0) = -sin(x) cos(y)
    v(x, y, 0) = cos(x) sin(y)

    Args:
        x, y: Coordinate arrays (assumed to be in [0, 2π])
        amplitude: Velocity amplitude

    Returns:
        Tuple of (u, v, params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - params: Dict with amplitude
    """
    X, Y = np.meshgrid(x, y)

    u = -amplitude * np.sin(X) * np.cos(Y)
    v = amplitude * np.cos(X) * np.sin(Y)

    params = {
        'amplitude': amplitude
    }

    return u, v, params


def shear_layer_ic(
    y_center: float,
    thickness: float,
    velocity_jump: float,
    perturbation_amplitude: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
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
        Tuple of (u, v, params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - params: Dict with y_center, thickness, velocity_jump, perturbation_amplitude
    """
    X, Y = np.meshgrid(x, y)

    # Base flow: u varies with y, v = 0
    u_base = (velocity_jump / 2) * np.tanh((Y - y_center) / thickness)

    # Add sinusoidal perturbation to trigger instability
    v_pert = perturbation_amplitude * np.sin(2 * np.pi * X / (x[-1] - x[0]))

    u = u_base
    v = v_pert

    params = {
        'y_center': y_center,
        'thickness': thickness,
        'velocity_jump': velocity_jump,
        'perturbation_amplitude': perturbation_amplitude
    }

    return u, v, params


def lamb_oseen_vortex_ic(
    center: Tuple[float, float],
    core_radius: float,
    circulation: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
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
        Tuple of (u, v, params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - params: Dict with center, core_radius, circulation
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

    params = {
        'center': center,
        'core_radius': core_radius,
        'circulation': circulation
    }

    return u, v, params


def dipole_vortex_ic(
    center: Tuple[float, float],
    separation: float,
    width: float,
    strength: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, list]:
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
        Tuple of (u, v, vortex_params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - vortex_params: List of dicts with actual vortex parameters
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
    y: np.ndarray,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate uniform flow with small random perturbations.

    Useful for studying transition to turbulence.

    Args:
        u_mean: Mean x-velocity
        v_mean: Mean y-velocity
        perturbation_amplitude: RMS amplitude of perturbations
        perturbation_wavelength: Dominant wavelength of perturbations
        x, y: Coordinate arrays
        seed: Random seed for reproducibility (None = truly random)

    Returns:
        Tuple of (u, v, params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - params: Dict with 'modes' list containing phase/wavenumber info
    """
    X, Y = np.meshgrid(x, y)

    # Random Fourier modes
    rng = np.random.RandomState(seed)

    k_pert = 2 * np.pi / perturbation_wavelength

    # Generate random phases
    n_modes = 5
    u_pert = np.zeros_like(X)
    v_pert = np.zeros_like(Y)

    modes = []
    for i in range(n_modes):
        phase_u = rng.uniform(0, 2 * np.pi)
        phase_v = rng.uniform(0, 2 * np.pi)
        kx = k_pert * rng.uniform(0.5, 1.5)
        ky = k_pert * rng.uniform(0.5, 1.5)

        u_pert += np.sin(kx * X + ky * Y + phase_u)
        v_pert += np.sin(kx * X + ky * Y + phase_v)

        modes.append({
            'kx': kx,
            'ky': ky,
            'phase_u': phase_u,
            'phase_v': phase_v
        })

    # Normalize perturbations
    u_pert = perturbation_amplitude * u_pert / np.sqrt(n_modes)
    v_pert = perturbation_amplitude * v_pert / np.sqrt(n_modes)

    u = u_mean + u_pert
    v = v_mean + v_pert

    params = {
        'u_mean': u_mean,
        'v_mean': v_mean,
        'perturbation_amplitude': perturbation_amplitude,
        'perturbation_wavelength': perturbation_wavelength,
        'modes': modes
    }

    return u, v, params


def random_vortex_soup_ic(
    n_vortices: int,
    strength_range: Tuple[float, float],
    width_range: Tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, list]:
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
        Tuple of (u, v, vortex_params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - vortex_params: List of dicts with actual parameters used
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


def gaussian_vortex_ic(
    n_gaussians: int,
    amplitude_range: Tuple[float, float],
    width_range: Tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate velocity field from sum of random Gaussian vorticity distributions.

    This is the primary IC generator for meta-learning: creates diverse swirling
    flow patterns by randomly placing multiple Gaussian vorticity bumps.

    The process:
    1. Sample N Gaussians with random (amplitude, width, x₀, y₀) parameters
    2. Sum as vorticity field: ω = Σ Aᵢ·exp(-rᵢ²/wᵢ²)
    3. Solve Poisson equation ∇²ψ = -ω for stream function
    4. Extract velocity: u = -∂ψ/∂y, v = ∂ψ/∂x

    This approach guarantees divergence-free velocity fields (∇·u = 0) by construction,
    making it suitable for generating diverse training data for meta-learning.

    Args:
        n_gaussians: Number of Gaussian vorticity components to sum
        amplitude_range: (min, max) for random amplitude sampling (vortex strength)
        width_range: (min, max) for random width sampling (vortex size)
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        seed: Random seed for reproducibility (None = truly random)

    Returns:
        Tuple of (u, v, vortex_params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - vortex_params: List of dicts with actual parameters used

    Example:
        >>> x = np.linspace(0, 2*np.pi, 64)
        >>> y = np.linspace(0, 2*np.pi, 64)
        >>> u, v, params = gaussian_vortex_ic(
        ...     n_gaussians=5,
        ...     amplitude_range=(-2.0, 2.0),
        ...     width_range=(0.2, 0.8),
        ...     x=x, y=y,
        ...     seed=42
        ... )
    """
    rng = np.random.RandomState(seed)

    # Domain bounds for random placement
    domain_x = (x[0], x[-1])
    domain_y = (y[0], y[-1])

    # Generate random parameters for each Gaussian
    vortex_params = []
    for _ in range(n_gaussians):
        center_x = rng.uniform(domain_x[0], domain_x[1])
        center_y = rng.uniform(domain_y[0], domain_y[1])
        amplitude = rng.uniform(amplitude_range[0], amplitude_range[1])
        width = rng.uniform(width_range[0], width_range[1])

        vortex_params.append({
            'center': (center_x, center_y),
            'width': width,
            'strength': amplitude
        })

    # Use existing multi_vortex infrastructure to compute velocity
    return multi_vortex_ic(vortex_params, x, y)


def gaussian_direct_ic(
    n_gaussians_u: int,
    n_gaussians_v: int,
    amplitude_range: Tuple[float, float],
    width_range: Tuple[float, float],
    x: np.ndarray,
    y: np.ndarray,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate velocity field by directly constructing u and v from independent Gaussian sums.

    Unlike the vorticity-based approach, this method independently constructs each velocity
    component using separate sets of Gaussian functions. This creates arbitrary flow patterns
    without the rotational structure imposed by vorticity fields.

    The resulting velocity field is NOT guaranteed to be divergence-free initially, but
    PhiFlow's pressure projection will enforce incompressibility at the first timestep.

    Args:
        n_gaussians_u: Number of Gaussians for u component
        n_gaussians_v: Number of Gaussians for v component
        amplitude_range: (min, max) for random amplitude sampling
        width_range: (min, max) for random width sampling
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        seed: Random seed for reproducibility (None = truly random)

    Returns:
        Tuple of (u, v, params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - params: Dict with 'u_gaussians' and 'v_gaussians' lists

    Example:
        >>> x = np.linspace(0, 2*np.pi, 64)
        >>> y = np.linspace(0, 2*np.pi, 64)
        >>> u, v, params = gaussian_direct_ic(
        ...     n_gaussians_u=5,
        ...     n_gaussians_v=3,
        ...     amplitude_range=(-2.0, 2.0),
        ...     width_range=(0.2, 0.8),
        ...     x=x, y=y,
        ...     seed=42
        ... )
    """
    rng = np.random.RandomState(seed)

    # Create 2D grid
    X, Y = np.meshgrid(x, y)

    # Domain bounds
    domain_x = (x[0], x[-1])
    domain_y = (y[0], y[-1])

    # Initialize velocity components
    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    # Track generated parameters
    u_gaussians = []
    v_gaussians = []

    # Build u component from independent Gaussians
    for _ in range(n_gaussians_u):
        center_x = rng.uniform(domain_x[0], domain_x[1])
        center_y = rng.uniform(domain_y[0], domain_y[1])
        amplitude = rng.uniform(amplitude_range[0], amplitude_range[1])
        width = rng.uniform(width_range[0], width_range[1])

        r_squared = (X - center_x)**2 + (Y - center_y)**2
        u += amplitude * np.exp(-r_squared / (2 * width**2))

        u_gaussians.append({
            'center': (center_x, center_y),
            'width': width,
            'amplitude': amplitude
        })

    # Build v component from different independent Gaussians
    for _ in range(n_gaussians_v):
        center_x = rng.uniform(domain_x[0], domain_x[1])
        center_y = rng.uniform(domain_y[0], domain_y[1])
        amplitude = rng.uniform(amplitude_range[0], amplitude_range[1])
        width = rng.uniform(width_range[0], width_range[1])

        r_squared = (X - center_x)**2 + (Y - center_y)**2
        v += amplitude * np.exp(-r_squared / (2 * width**2))

        v_gaussians.append({
            'center': (center_x, center_y),
            'width': width,
            'amplitude': amplitude
        })

    params = {
        'u_gaussians': u_gaussians,
        'v_gaussians': v_gaussians
    }

    return u, v, params


def gaussian_hybrid_ic(
    n_gaussians_vorticity: int,
    n_gaussians_u: int,
    n_gaussians_v: int,
    amplitude_range: Tuple[float, float],
    width_range: Tuple[float, float],
    alpha: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate hybrid velocity field combining vorticity-based and direct construction.

    Creates a weighted combination of:
    1. Vorticity-based component (swirling, divergence-free)
    2. Direct velocity component (arbitrary directions, not divergence-free)

    The mixing formula is:
        u_final = alpha * u_vorticity + beta * u_direct
        v_final = alpha * v_vorticity + beta * v_direct

    where alpha, beta ∈ (-∞, ∞) provide full control over the balance.

    Args:
        n_gaussians_vorticity: Number of Gaussians for vorticity field
        n_gaussians_u: Number of Gaussians for direct u component
        n_gaussians_v: Number of Gaussians for direct v component
        amplitude_range: (min, max) for random amplitude sampling
        width_range: (min, max) for random width sampling
        alpha: Weight for vorticity component (unbounded)
        beta: Weight for direct component (unbounded)
        x: 1D array of x-coordinates
        y: 1D array of y-coordinates
        seed: Random seed for reproducibility (None = truly random)

    Returns:
        Tuple of (u, v, params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - params: Dict with 'vorticity_gaussians', 'direct_params', 'alpha', 'beta'

    Examples:
        Pure vorticity (swirling only):
        >>> u, v, p = gaussian_hybrid_ic(..., alpha=1.0, beta=0.0, ...)

        Equal contribution:
        >>> u, v, p = gaussian_hybrid_ic(..., alpha=1.0, beta=1.0, ...)

        Mostly vorticity with direct perturbations:
        >>> u, v, p = gaussian_hybrid_ic(..., alpha=0.7, beta=0.3, ...)

        Inverted vorticity:
        >>> u, v, p = gaussian_hybrid_ic(..., alpha=-1.0, beta=1.0, ...)
    """
    # Generate vorticity-based component (swirling, divergence-free)
    u_vort, v_vort, vorticity_params = gaussian_vortex_ic(
        n_gaussians=n_gaussians_vorticity,
        amplitude_range=amplitude_range,
        width_range=width_range,
        x=x, y=y,
        seed=seed
    )

    # Generate direct component (arbitrary directions, not divergence-free)
    # Use seed offset to ensure decorrelation when seed is specified
    seed_direct = (seed + 1000) if seed is not None else None
    u_direct, v_direct, direct_params = gaussian_direct_ic(
        n_gaussians_u=n_gaussians_u,
        n_gaussians_v=n_gaussians_v,
        amplitude_range=amplitude_range,
        width_range=width_range,
        x=x, y=y,
        seed=seed_direct
    )

    # Weighted combination
    u = alpha * u_vort + beta * u_direct
    v = alpha * v_vort + beta * v_direct

    params = {
        'vorticity_gaussians': vorticity_params,
        'direct_params': direct_params,
        'alpha': alpha,
        'beta': beta
    }

    return u, v, params


def von_karman_street_ic(
    n_vortices: int,
    spacing: float,
    offset: float,
    width: float,
    strength: float,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, list]:
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
        Tuple of (u, v, vortex_params) where:
        - u: x-component of velocity, shape (ny, nx)
        - v: y-component of velocity, shape (ny, nx)
        - vortex_params: List of dicts with actual vortex parameters
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
