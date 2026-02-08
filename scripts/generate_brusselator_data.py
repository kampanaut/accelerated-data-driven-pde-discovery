"""
Generate Brusselator training data for PDE discovery.

This script:
1. Loads initial condition configuration from YAML file
2. Solves Brusselator equations for each IC
3. Computes spatial and temporal derivatives
4. Formats data as training samples for the N network
5. Saves data and visualizations

Usage:
    python scripts/generate_brusselator_data.py --config configs/brusselator_train.yaml
    python scripts/generate_brusselator_data.py --config configs/brusselator_train.yaml --gpu --workers 8
"""

import sys
import os
from pathlib import Path

# Default to CPU for stability (PhiFlow iterative solvers diverge more on GPU)
# Use --gpu to opt into GPU execution
if '--gpu' in sys.argv:
    sys.argv.remove('--gpu')
    os.environ['JAX_PLATFORMS'] = 'cuda'
elif os.environ.get('JAX_PLATFORMS') != 'cuda':
    # Only default to CPU if parent didn't already set GPU
    os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import argparse
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.brusselator import solve_brusselator
from src.data.derivatives import spectral_spatial_derivatives as spatial_derivatives
from src.data.initial_conditions_brusselator import create_brusselator_ic, compute_turing_threshold


def generate_training_samples(
    concentration_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    D_A: float,
    D_B: float,
    k1: float,
    k2: float,
) -> dict:
    """
    Convert concentration snapshots to training samples with derivatives.

    Per the "dead inputs" approach (experiment_bible.md §10.2), we compute
    all 10 derivatives to match NS network architecture, even though
    Brusselator only uses 6 (concentrations + second derivatives).

    Features: A, B, A_x, A_y, A_xx, A_yy, B_x, B_y, B_xx, B_yy (10 inputs)
    Targets: A_t, B_t (2 outputs)

    Targets are computed from the PDE RHS using the same spatial derivatives
    stored as features, NOT from central difference in time. This ensures
    features and targets are self-consistent. See docs/temporal_derivative_bias.md.

    Args:
        concentration_history: List of (A, B) tuples at different times
        times: Time values for each snapshot
        x, y: Coordinate arrays
        D_A, D_B: Diffusion coefficients
        k1, k2: Reaction rate constants

    Returns:
        Dict containing flattened training data
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    print(f"Computing derivatives...")
    print(f"  Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    print(f"  Targets: PDE RHS (D_A={D_A:.4f}, D_B={D_B:.4f}, k1={k1:.4f}, k2={k2:.4f})")

    # Initialize training data dict
    # Using u/v naming to match NS for network compatibility
    # A → u, B → v
    training_data = {
        'u': [],      # A concentration
        'v': [],      # B concentration
        'u_x': [],    # A_x (first derivative - "dead" for Brusselator)
        'u_y': [],    # A_y (first derivative - "dead" for Brusselator)
        'u_xx': [],   # A_xx (active for Brusselator)
        'u_yy': [],   # A_yy (active for Brusselator)
        'v_x': [],    # B_x (first derivative - "dead" for Brusselator)
        'v_y': [],    # B_y (first derivative - "dead" for Brusselator)
        'v_xx': [],   # B_xx (active for Brusselator)
        'v_yy': [],   # B_yy (active for Brusselator)
        'u_t': [],    # A_t (from PDE RHS)
        'v_t': [],    # B_t (from PDE RHS)
        't': [],
        'x_coord': [],
        'y_coord': [],
    }

    # Process all snapshots (no loss of first/last — no temporal central difference)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    for i, (A, B) in enumerate(concentration_history):
        t = times[i]

        # Spatial derivatives for both fields
        A_x, A_y, A_xx, A_yy = spatial_derivatives(A, dx, dy)
        B_x, B_y, B_xx, B_yy = spatial_derivatives(B, dx, dy)

        # Temporal derivatives from PDE RHS (self-consistent with stored spatial derivatives)
        A_t = D_A * (A_xx + A_yy) + k1 - (k2 + 1) * A + A**2 * B
        B_t = D_B * (B_xx + B_yy) + k2 * A - A**2 * B

        # Flatten spatial grids and append
        training_data['u'].append(A.flatten())
        training_data['v'].append(B.flatten())
        training_data['u_x'].append(A_x.flatten())
        training_data['u_y'].append(A_y.flatten())
        training_data['u_xx'].append(A_xx.flatten())
        training_data['u_yy'].append(A_yy.flatten())
        training_data['v_x'].append(B_x.flatten())
        training_data['v_y'].append(B_y.flatten())
        training_data['v_xx'].append(B_xx.flatten())
        training_data['v_yy'].append(B_yy.flatten())
        training_data['u_t'].append(A_t.flatten())
        training_data['v_t'].append(B_t.flatten())
        training_data['x_coord'].append(x_flat)
        training_data['y_coord'].append(y_flat)
        training_data['t'].append(np.full(X.size, t))

    # Stack all timesteps together
    for key in training_data:
        training_data[key] = np.concatenate(training_data[key])

    n_samples = len(training_data['u'])
    n_snapshots = len(concentration_history)
    print(f"  Generated {n_samples:,} training samples")
    print(f"  ({n_snapshots} timesteps × {A.size} spatial points)")

    return training_data


def generate_fourier_data(
    concentration_history: list,
    times: np.ndarray,
) -> dict:
    """
    Convert concentration snapshots to Fourier coefficient arrays.

    FFTs each snapshot and stacks into (n_snapshots, ny, nx) complex128 arrays.
    No derivative computation — derivatives are computed on-the-fly during
    training via wavenumber multiplication.

    Args:
        concentration_history: List of (A, B) tuples at different times
        times: Time values for each snapshot

    Returns:
        Dict with keys: A_hat, B_hat (complex128), times (float64)
    """
    n_snapshots = len(concentration_history)
    ny, nx = concentration_history[0][0].shape

    A_hat_stack = np.empty((n_snapshots, ny, nx), dtype=np.complex128)
    B_hat_stack = np.empty((n_snapshots, ny, nx), dtype=np.complex128)

    for i, (A, B) in enumerate(concentration_history):
        A_hat_stack[i] = np.fft.fft2(A)
        B_hat_stack[i] = np.fft.fft2(B)

    print(f"  FFT'd {n_snapshots} snapshots, shape ({ny}, {nx})")

    return {
        'A_hat': A_hat_stack,
        'B_hat': B_hat_stack,
        'times': times,
    }


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_metadata_txt(output_path: Path, ic_config: dict, simulation_params: dict,
                      training_data: dict, generated_params: dict = None,
                      success: bool = True, error_msg: str = None):
    """
    Save human-readable metadata alongside .npz file.
    Shows exact parameters used for reproducibility/debugging.
    """
    txt_path = output_path.with_suffix('.txt')

    with open(txt_path, 'w') as f:
        f.write("Brusselator Simulation Metadata\n")
        f.write("=" * 60 + "\n")
        f.write(f"Task Name: {output_path.stem}\n")
        f.write(f"IC Type: {ic_config.get('type', 'unknown')}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
        if not success and error_msg:
            f.write(f"Error: {error_msg}\n")
        f.write("\n")

        # IC parameters (from config)
        f.write("Initial Condition Parameters (from config):\n")
        for key, value in ic_config.items():
            if key not in ['type', 'name']:
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Generated parameters (actual values used)
        if generated_params:
            f.write("Generated Parameters (actual values):\n")
            _write_generated_params(f, generated_params)
            f.write("\n")

        # Simulation parameters
        f.write("Simulation Parameters:\n")
        f.write(f"  D_A (diffusion A): {simulation_params['D_A']:.6f}\n")
        f.write(f"  D_B (diffusion B): {simulation_params['D_B']:.6f}\n")
        f.write(f"  k₁ (reaction): {simulation_params['k1']:.6f}\n")
        f.write(f"  k₂ (reaction): {simulation_params['k2']:.6f}\n")
        f.write(f"  Steady state: A* = {simulation_params['k1']:.4f}, B* = {simulation_params['k2']/simulation_params['k1']:.4f}\n")
        f.write(f"  Domain: {simulation_params['domain_size']}\n")
        f.write(f"  Resolution: {simulation_params['resolution']}\n")
        f.write(f"  Time range: [0, {simulation_params['t_end']}]\n")
        f.write(f"  Time step: dt={simulation_params['dt']}\n")
        if 'save_interval' in simulation_params:
            f.write(f"  Save interval: {simulation_params['save_interval']}\n")
        f.write("\n")

        if success and training_data:
            # Output statistics
            f.write("Output Statistics:\n")
            f.write(f"  Total samples: {len(training_data['u']):,}\n")
            f.write(f"  Features: 10 (A, B, spatial derivatives - NS-compatible format)\n")
            f.write(f"  Targets: 2 (A_t, B_t)\n")
            f.write("\n")

            # Data ranges (using A/B naming for clarity)
            f.write("Data Ranges:\n")
            for key, label in [('u', 'A'), ('v', 'B'), ('u_t', 'A_t'), ('v_t', 'B_t')]:
                if key in training_data:
                    values = training_data[key]
                    f.write(f"  {label:4s}: [{values.min():>8.4f}, {values.max():>8.4f}]\n")
            f.write("\n")

            # Check for potential issues
            f.write("Data Quality Checks:\n")
            has_nan = any(np.any(np.isnan(training_data[k])) for k in ['u', 'v', 'u_t', 'v_t'])
            has_inf = any(np.any(np.isinf(training_data[k])) for k in ['u', 'v', 'u_t', 'v_t'])
            f.write(f"  NaN values: {'DETECTED' if has_nan else 'None'}\n")
            f.write(f"  Inf values: {'DETECTED' if has_inf else 'None'}\n")


def _write_generated_params(f, params, indent=2):
    """Helper to write generated parameters recursively."""
    prefix = " " * indent
    if isinstance(params, list):
        for i, item in enumerate(params):
            if isinstance(item, dict):
                f.write(f"{prefix}[{i}]:\n")
                _write_generated_params(f, item, indent + 4)
            else:
                f.write(f"{prefix}[{i}]: {item}\n")
    elif isinstance(params, dict):
        for key, value in params.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 60:
                f.write(f"{prefix}{key}:\n")
                _write_generated_params(f, value, indent + 4)
            elif isinstance(value, tuple) and len(value) == 2:
                f.write(f"{prefix}{key}: ({value[0]:.4f}, {value[1]:.4f})\n")
            elif isinstance(value, float):
                f.write(f"{prefix}{key}: {value:.6f}\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")
    else:
        f.write(f"{prefix}{params}\n")


def save_brusselator_evolution(
    concentration_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    n_snapshots: int = 4
):
    """
    Save a multi-panel figure showing Brusselator evolution over time.

    Creates a 4×n_snapshots grid:
    - Row 1: A concentration (2D contour)
    - Row 2: B concentration (2D contour)
    - Row 3: A concentration (3D surface)
    - Row 4: B concentration (3D surface)

    Args:
        concentration_history: List of (A, B) tuples at different times
        times: Array of time values
        x, y: Coordinate arrays
        output_path: Path to save the figure
        n_snapshots: Number of snapshots to show
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    indices = np.linspace(0, len(concentration_history) - 1, n_snapshots, dtype=int)

    fig = plt.figure(figsize=(5 * n_snapshots, 20))

    # Prepare grid
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y)
    else:
        x_grid, y_grid = x, y

    # Get global min/max for consistent colorbars
    all_A = [concentration_history[i][0] for i in indices]
    all_B = [concentration_history[i][1] for i in indices]
    A_min, A_max = min(a.min() for a in all_A), max(a.max() for a in all_A)
    B_min, B_max = min(b.min() for b in all_B), max(b.max() for b in all_B)

    for col, idx in enumerate(indices):
        A, B = concentration_history[idx]
        t = times[idx]

        # Row 1: A concentration (2D)
        ax_A_2d = plt.subplot(4, n_snapshots, col + 1)
        contour_A = ax_A_2d.contourf(x_grid, y_grid, A, levels=20, cmap='YlOrRd', vmin=A_min, vmax=A_max)
        plt.colorbar(contour_A, ax=ax_A_2d, fraction=0.046, pad=0.04)
        ax_A_2d.set_title(f't={t:.3f}')
        ax_A_2d.set_aspect('equal')

        # Row 2: B concentration (2D)
        ax_B_2d = plt.subplot(4, n_snapshots, n_snapshots + col + 1)
        contour_B = ax_B_2d.contourf(x_grid, y_grid, B, levels=20, cmap='YlGnBu', vmin=B_min, vmax=B_max)
        plt.colorbar(contour_B, ax=ax_B_2d, fraction=0.046, pad=0.04)
        ax_B_2d.set_aspect('equal')

        # Row 3: A concentration (3D surface)
        ax_A_3d = plt.subplot(4, n_snapshots, 2 * n_snapshots + col + 1, projection='3d')
        surf_A = ax_A_3d.plot_surface(
            x_grid, y_grid, A,
            cmap='YlOrRd',
            linewidth=0,
            antialiased=True,
            alpha=0.9
        )
        ax_A_3d.set_xlabel('x')
        ax_A_3d.set_ylabel('y')
        ax_A_3d.set_zlabel('A')
        ax_A_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_A, ax=ax_A_3d, fraction=0.03, pad=0.1, shrink=0.5)

        # Row 4: B concentration (3D surface)
        ax_B_3d = plt.subplot(4, n_snapshots, 3 * n_snapshots + col + 1, projection='3d')
        surf_B = ax_B_3d.plot_surface(
            x_grid, y_grid, B,
            cmap='YlGnBu',
            linewidth=0,
            antialiased=True,
            alpha=0.9
        )
        ax_B_3d.set_xlabel('x')
        ax_B_3d.set_ylabel('y')
        ax_B_3d.set_zlabel('B')
        ax_B_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_B, ax=ax_B_3d, fraction=0.03, pad=0.1, shrink=0.5)

    # Add row labels
    fig.text(0.02, 0.88, 'A conc (2D)', va='center', rotation='vertical', fontsize=11, weight='bold')
    fig.text(0.02, 0.65, 'B conc (2D)', va='center', rotation='vertical', fontsize=11, weight='bold')
    fig.text(0.02, 0.42, 'A conc (3D)', va='center', rotation='vertical', fontsize=11, weight='bold')
    fig.text(0.02, 0.18, 'B conc (3D)', va='center', rotation='vertical', fontsize=11, weight='bold')

    fig.suptitle('Brusselator Evolution', fontsize=16, y=0.99)
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved evolution visualization to {output_path}")


def process_single_ic(args_tuple):
    """
    Worker function for multiprocessing Pool.

    Processes a single IC configuration, including retry logic for divergence.

    Args:
        args_tuple: (ic_config, simulation_params, data_dir, ic_idx, total_count, x, y)

    Returns:
        (status, ic_name, error_msg, retry_count)
        status: 'success', 'skipped', or 'failed'
    """
    *core_args, fourier_mode = args_tuple
    ic_config, simulation_params, data_dir, ic_idx, total_count, x, y = core_args

    ic_name = ic_config.get('name', f'ic_{ic_idx}')

    # Check if already generated
    suffix = "_fourier" if fourier_mode else ""
    output_file = Path(data_dir) / f"{ic_name}{suffix}.npz"
    if output_file.exists():
        return ('skipped', ic_name, None, 0)

    # Check for task-specific parameter overrides
    task_sim_params = simulation_params.copy()

    # Handle parameter sampling from ranges
    # Single RNG per task — draws are sequential so D_A, D_B, k1, k2 get distinct values
    param_rng = np.random.default_rng(ic_config.get('seed'))
    for param in ['D_A', 'D_B', 'k1', 'k2']:
        raw_val = ic_config.get(param, task_sim_params[param])
        if isinstance(raw_val, list):
            if len(raw_val) != 2:
                return ('failed', ic_name, f"{param} as list must have 2 elements, got {len(raw_val)}", 0)
            task_sim_params[param] = param_rng.uniform(raw_val[0], raw_val[1])
        elif raw_val != task_sim_params[param]:
            task_sim_params[param] = raw_val

    # k2_delta: sample k2 relative to Turing threshold instead of absolute range
    k2_delta_range = ic_config.get('k2_delta')
    if k2_delta_range is not None:
        k2_c = compute_turing_threshold(
            k1=task_sim_params['k1'],
            D_A=task_sim_params['D_A'],
            D_B=task_sim_params['D_B'],
        )
        delta = param_rng.uniform(k2_delta_range[0], k2_delta_range[1])
        task_sim_params['k2'] = k2_c + delta

    # Inject k1, k2 into IC config for steady state calculation
    ic_config_with_params = ic_config.copy()
    ic_config_with_params['k1'] = task_sim_params['k1']
    ic_config_with_params['k2'] = task_sim_params['k2']

    # Retry loop for divergence
    max_retries = 800
    base_seed = ic_config.get('seed', None)

    for attempt in range(max_retries):
        ic_config_attempt = ic_config_with_params.copy()
        if base_seed is not None:
            ic_config_attempt['seed'] = base_seed + attempt * 1000

        try:
            # Create initial condition
            A_init, B_init, generated_params = create_brusselator_ic(ic_config_attempt, x, y)

            # Solve Brusselator
            concentration_history, times, x_result, y_result = solve_brusselator(
                initial_concentration=(A_init, B_init),
                D_A=task_sim_params['D_A'],
                D_B=task_sim_params['D_B'],
                k1=task_sim_params['k1'],
                k2=task_sim_params['k2'],
                domain_size=task_sim_params['domain_size'],
                t_end=task_sim_params['t_end'],
                dt=task_sim_params['dt'],
                save_interval=task_sim_params.get('save_interval')
            )

            # Validate raw concentrations for divergence
            max_magnitude = 1e6
            last_A, last_B = concentration_history[-1]
            for label, arr in [('A', last_A), ('B', last_B)]:
                max_val = np.abs(arr).max()
                if max_val > max_magnitude or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise ValueError(f"Silent divergence: {label} has max magnitude {max_val:.2e}")

            ic_config_to_save = ic_config.copy()
            ic_config_to_save['k1_used'] = task_sim_params['k1']
            ic_config_to_save['k2_used'] = task_sim_params['k2']
            ic_config_to_save['D_A_used'] = task_sim_params['D_A']
            ic_config_to_save['D_B_used'] = task_sim_params['D_B']
            ic_config_to_save['seed_used'] = ic_config_attempt.get('seed')
            ic_config_to_save['retry_attempt'] = attempt
            if k2_delta_range is not None:
                ic_config_to_save['k2_c'] = k2_c
                ic_config_to_save['k2_delta_used'] = task_sim_params['k2'] - k2_c

            if fourier_mode:
                fourier_data = generate_fourier_data(concentration_history, times)
                np.savez(
                    output_file,
                    **fourier_data,
                    ic_config=ic_config_to_save,
                    simulation_params=task_sim_params,
                )
            else:
                training_data = generate_training_samples(
                    concentration_history, times, x_result, y_result,
                    D_A=task_sim_params['D_A'], D_B=task_sim_params['D_B'],
                    k1=task_sim_params['k1'], k2=task_sim_params['k2'],
                )

                # Validate training data
                for key in ['u', 'v', 'u_t', 'v_t']:
                    if key in training_data:
                        max_val = np.abs(training_data[key]).max()
                        if max_val > max_magnitude or np.any(np.isnan(training_data[key])) or np.any(np.isinf(training_data[key])):
                            raise ValueError(f"Silent divergence: {key} has max magnitude {max_val:.2e}")

                np.savez(
                    output_file,
                    **training_data,
                    ic_config=ic_config_to_save,
                    simulation_params=task_sim_params,
                    x=x_result,
                    y=y_result,
                )

                # Save metadata
                save_metadata_txt(output_file, ic_config_to_save, task_sim_params, training_data, generated_params, success=True)

            # Generate visualization
            vis_file = Path(data_dir) / f"{ic_name}_evolution.png"
            save_brusselator_evolution(concentration_history, times, x_result, y_result, str(vis_file), n_snapshots=4)

            return ('success', ic_name, None, attempt)

        except ValueError as e:
            if "divergence" in str(e).lower() and attempt < max_retries - 1:
                continue
            if attempt >= max_retries - 1:
                return ('failed', ic_name, str(e), max_retries)
            continue

        except Exception as e:
            # Check if it's a divergence-related error
            error_str = str(e).lower()
            if any(kw in error_str for kw in ['diverge', 'nan', 'inf', 'overflow']) and attempt < max_retries - 1:
                continue
            return ('failed', ic_name, f"{type(e).__name__}: {str(e)}", attempt)

    return ('failed', ic_name, f"Exhausted {max_retries} retries", max_retries)


def main():
    """Main data generation workflow."""
    parser = argparse.ArgumentParser(description='Generate Brusselator training data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1 = serial)')
    parser.add_argument('--fourier', action='store_true',
                        help='Save Fourier coefficients instead of grid derivatives')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("=" * 60)
    print("Brusselator Data Generation for PDE Discovery")
    print("=" * 60)
    print(f"Config file: {config_path}")

    # Extract simulation parameters
    sim_params = config['simulation']
    simulation_params = {
        'D_A': sim_params['D_A'],
        'D_B': sim_params['D_B'],
        'k1': sim_params['k1'],
        'k2': sim_params['k2'],
        'domain_size': tuple(sim_params['domain_size']),
        'resolution': tuple(sim_params['resolution']),
        't_end': sim_params['t_end'],
        'dt': sim_params['dt'],
        'save_interval': sim_params['save_interval'],
    }

    print(f"\nSimulation parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")
    print(f"  Steady state: A* = {simulation_params['k1']:.4f}, B* = {simulation_params['k2']/simulation_params['k1']:.4f}")

    # Extract IC configurations
    ic_configs = config['initial_conditions']
    print(f"\nInitial conditions: {len(ic_configs)} configurations")

    # Create output directory
    sim_name = config.get('simulation', {}).get('name') or config.get('output_dir', 'brusselator')
    base_dir = config.get('output', {}).get('base_dir', 'data/datasets')
    data_dir = Path(__file__).parent.parent / base_dir / sim_name
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {data_dir}")

    # Setup coordinate arrays
    domain_size = simulation_params['domain_size']
    resolution = simulation_params['resolution']
    x = np.linspace(0, domain_size[0], resolution[1])
    y = np.linspace(0, domain_size[1], resolution[0])

    # Build work items for processing
    work_items = [
        (ic_config, simulation_params, str(data_dir), ic_idx, len(ic_configs), x, y, args.fourier)
        for ic_idx, ic_config in enumerate(ic_configs)
    ]

    # Process ICs (parallel or serial)
    if args.workers > 1:
        # Parallel execution with compact progress output
        # Use 'spawn' context to avoid JAX fork() deadlock
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')

        print(f"\nUsing {args.workers} parallel workers")

        results = []
        with ctx.Pool(args.workers) as pool:
            for result in pool.imap_unordered(process_single_ic, work_items):
                status, name, error, retries = result
                results.append(result)

                # Progress reporting
                done = len(results)
                if status == 'success':
                    retry_info = f" (after {retries} retries)" if retries > 0 else ""
                    print(f"[{done}/{len(ic_configs)}] {name}: ✓ SUCCESS{retry_info}")
                elif status == 'skipped':
                    print(f"[{done}/{len(ic_configs)}] {name}: ⊘ SKIPPED (exists)")
                else:
                    print(f"[{done}/{len(ic_configs)}] {name}: ✗ FAILED - {error}")

        # Tally results
        successful = sum(1 for r in results if r[0] in ('success', 'skipped'))
        skipped_existing = sum(1 for r in results if r[0] == 'skipped')
        failed_tasks = [(r[1], r[2]) for r in results if r[0] == 'failed']

    else:
        # Serial execution with original verbose output
        successful = 0
        failed_tasks = []
        skipped_existing = 0

        for ic_idx, ic_config in enumerate(ic_configs):
            ic_name = ic_config.get('name', f'ic_{ic_idx}')

            # Check if already generated
            suffix = "_fourier" if args.fourier else ""
            output_file = data_dir / f"{ic_name}{suffix}.npz"
            if output_file.exists():
                print(f"[{ic_idx + 1}/{len(ic_configs)}] {ic_name}: already exists, skipping")
                skipped_existing += 1
                successful += 1
                continue

            print(f"\n{'-' * 60}")
            print(f"IC {ic_idx + 1}/{len(ic_configs)}: {ic_name} ({ic_config['type']})")
            print(f"{'-' * 60}")

            # Check for task-specific parameter overrides
            task_sim_params = simulation_params.copy()

            # Handle parameter sampling from ranges
            # Single RNG per task — draws are sequential so D_A, D_B, k1, k2 get distinct values
            param_rng = np.random.default_rng(ic_config.get('seed'))
            for param in ['D_A', 'D_B', 'k1', 'k2']:
                raw_val = ic_config.get(param, task_sim_params[param])
                if isinstance(raw_val, list):
                    if len(raw_val) != 2:
                        raise ValueError(
                            f"Task '{ic_name}': {param} as list must have exactly 2 elements [min, max], "
                            f"got {len(raw_val)} elements: {raw_val}"
                        )
                    task_sim_params[param] = param_rng.uniform(raw_val[0], raw_val[1])
                    print(f"Sampled {param} = {task_sim_params[param]:.6f} from range {raw_val}")
                elif raw_val != task_sim_params[param]:
                    task_sim_params[param] = raw_val

            # k2_delta: sample k2 relative to Turing threshold
            k2_delta_range = ic_config.get('k2_delta')
            if k2_delta_range is not None:
                k2_c = compute_turing_threshold(
                    k1=task_sim_params['k1'],
                    D_A=task_sim_params['D_A'],
                    D_B=task_sim_params['D_B'],
                )
                delta = param_rng.uniform(k2_delta_range[0], k2_delta_range[1])
                task_sim_params['k2'] = k2_c + delta
                print(f"k2_delta mode: k2_c={k2_c:.4f}, delta={delta:.4f}, k2={task_sim_params['k2']:.4f}")

            # Inject k1, k2 into IC config for steady state calculation
            ic_config_with_params = ic_config.copy()
            ic_config_with_params['k1'] = task_sim_params['k1']
            ic_config_with_params['k2'] = task_sim_params['k2']

            # Retry loop for divergence
            max_retries = 800
            base_seed = ic_config.get('seed', None)
            task_succeeded = False

            for attempt in range(max_retries):
                ic_config_attempt = ic_config_with_params.copy()
                if base_seed is not None:
                    ic_config_attempt['seed'] = base_seed + attempt * 1000
                    if attempt > 0:
                        print(f"  Retry {attempt}/{max_retries-1} with seed {ic_config_attempt['seed']}")

                try:
                    # Create initial condition
                    A_init, B_init, generated_params = create_brusselator_ic(ic_config_attempt, x, y)

                    # Solve Brusselator
                    concentration_history, times, x_result, y_result = solve_brusselator(
                        initial_concentration=(A_init, B_init),
                        D_A=task_sim_params['D_A'],
                        D_B=task_sim_params['D_B'],
                        k1=task_sim_params['k1'],
                        k2=task_sim_params['k2'],
                        domain_size=task_sim_params['domain_size'],
                        t_end=task_sim_params['t_end'],
                        dt=task_sim_params['dt'],
                        save_interval=task_sim_params.get('save_interval')
                    )

                    # Validate raw concentrations for divergence
                    max_magnitude = 1e6
                    last_A, last_B = concentration_history[-1]
                    for label, arr in [('A', last_A), ('B', last_B)]:
                        max_val = np.abs(arr).max()
                        if max_val > max_magnitude or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                            raise ValueError(f"Silent divergence: {label} has max magnitude {max_val:.2e}")

                    ic_config_to_save = ic_config.copy()
                    ic_config_to_save['k1_used'] = task_sim_params['k1']
                    ic_config_to_save['k2_used'] = task_sim_params['k2']
                    ic_config_to_save['D_A_used'] = task_sim_params['D_A']
                    ic_config_to_save['D_B_used'] = task_sim_params['D_B']
                    ic_config_to_save['seed_used'] = ic_config_attempt.get('seed')
                    ic_config_to_save['retry_attempt'] = attempt
                    if k2_delta_range is not None:
                        ic_config_to_save['k2_c'] = k2_c
                        ic_config_to_save['k2_delta_used'] = task_sim_params['k2'] - k2_c

                    if args.fourier:
                        fourier_data = generate_fourier_data(concentration_history, times)
                        np.savez(
                            output_file,
                            **fourier_data,
                            ic_config=ic_config_to_save,
                            simulation_params=task_sim_params,
                        )
                        print(f"\nSaved Fourier data to {output_file}")
                    else:
                        training_data = generate_training_samples(
                            concentration_history, times, x_result, y_result,
                            D_A=task_sim_params['D_A'], D_B=task_sim_params['D_B'],
                            k1=task_sim_params['k1'], k2=task_sim_params['k2'],
                        )

                        # Validate training data
                        for key in ['u', 'v', 'u_t', 'v_t']:
                            if key in training_data:
                                max_val = np.abs(training_data[key]).max()
                                if max_val > max_magnitude or np.any(np.isnan(training_data[key])) or np.any(np.isinf(training_data[key])):
                                    raise ValueError(f"Silent divergence: {key} has max magnitude {max_val:.2e}")

                        np.savez(
                            output_file,
                            **training_data,
                            ic_config=ic_config_to_save,
                            simulation_params=task_sim_params,
                            x=x_result,
                            y=y_result,
                        )
                        print(f"\nSaved training data to {output_file}")

                        # Save metadata
                        save_metadata_txt(output_file, ic_config_to_save, task_sim_params, training_data, generated_params, success=True)

                        # Print sample data
                        print(f"\nSample data point (first sample):")
                        print(f"  A={training_data['u'][0]:.4f}, B={training_data['v'][0]:.4f}")
                        print(f"  A_xx={training_data['u_xx'][0]:.4f}, A_yy={training_data['u_yy'][0]:.4f}")
                        print(f"  A_t={training_data['u_t'][0]:.4f}, B_t={training_data['v_t'][0]:.4f}")
                        print(f"  t={training_data['t'][0]:.4f}, x={training_data['x_coord'][0]:.4f}, y={training_data['y_coord'][0]:.4f}")

                    if attempt > 0:
                        print(f"  (succeeded after {attempt} retries)")

                    # Generate visualization
                    vis_file = data_dir / f"{ic_name}_evolution.png"
                    save_brusselator_evolution(concentration_history, times, x_result, y_result, str(vis_file), n_snapshots=4)

                    successful += 1
                    break  # Success, exit retry loop

                except ValueError as e:
                    if "divergence" in str(e).lower() and attempt < max_retries - 1:
                        print(f"  ⚠️  {str(e)}, will retry...")
                        continue
                    if attempt >= max_retries - 1:
                        print(f"\n⚠️  SKIPPED: {ic_name}")
                        print(f"    Reason: {str(e)}")
                        print(f"    (failed all {max_retries} attempts)")
                        failed_tasks.append((ic_name, str(e)))
                        break
                    continue

                except Exception as e:
                    error_str = str(e).lower()
                    if any(kw in error_str for kw in ['diverge', 'nan', 'inf', 'overflow']) and attempt < max_retries - 1:
                        print(f"  ⚠️  {type(e).__name__}: {str(e)}, will retry...")
                        continue
                    print(f"\n⚠️  SKIPPED: {ic_name}")
                    print(f"    Error: {type(e).__name__}: {str(e)}")
                    failed_tasks.append((ic_name, str(e)))
                    break

    print(f"\n{'=' * 60}")
    print("Data generation complete!")
    print(f"{'=' * 60}")
    print(f"\nSummary:")
    print(f"  Total tasks: {len(ic_configs)}")
    print(f"  ✓ Successful: {successful} ({skipped_existing} already existed)")
    print(f"  ✗ Failed: {len(failed_tasks)}")
    print(f"\nOutput directory: {data_dir}")

    if failed_tasks:
        print(f"\n⚠️  Failed tasks:")
        for name, error in failed_tasks:
            print(f"    - {name}: {error}")

    if successful > 0:
        print(f"\nEach successful dataset contains:")
        print(f"  - Features: u(A), v(B), derivatives (10 inputs, NS-compatible format)")
        print(f"  - Targets: u_t(A_t), v_t(B_t) (2 outputs)")
        print(f"  - Metadata: t, x, y coordinates + .txt file with parameters")


if __name__ == '__main__':
    main()
