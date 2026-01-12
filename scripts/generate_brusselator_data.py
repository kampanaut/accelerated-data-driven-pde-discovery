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
"""

import numpy as np
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.brusselator import solve_brusselator
from src.data.derivatives import spatial_derivatives, temporal_derivative
from src.data.initial_conditions_brusselator import create_brusselator_ic


def generate_training_samples(
    concentration_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> dict:
    """
    Convert concentration snapshots to training samples with derivatives.

    Per the "dead inputs" approach (experiment_bible.md §10.2), we compute
    all 10 derivatives to match NS network architecture, even though
    Brusselator only uses 6 (concentrations + second derivatives).

    Features: A, B, A_x, A_y, A_xx, A_yy, B_x, B_y, B_xx, B_yy (10 inputs)
    Targets: A_t, B_t (2 outputs)

    Args:
        concentration_history: List of (A, B) tuples at different times
        times: Time values for each snapshot
        x, y: Coordinate arrays

    Returns:
        Dict containing flattened training data
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = times[1] - times[0]

    print(f"Computing derivatives...")
    print(f"  Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    print(f"  Time spacing: dt={dt:.4f}")

    # Separate A and B histories
    A_history = [conc[0] for conc in concentration_history]
    B_history = [conc[1] for conc in concentration_history]

    # Compute temporal derivatives (loses first and last timesteps)
    A_t = temporal_derivative(A_history, dt)
    B_t = temporal_derivative(B_history, dt)

    print(f"  Temporal derivatives shape: {A_t.shape}")

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
        'u_t': [],    # A_t
        'v_t': [],    # B_t
        't': [],
        'x_coord': [],
        'y_coord': [],
    }

    # Process snapshots with valid time derivatives (indices 1 to n-2)
    for i in range(len(A_t)):
        snapshot_idx = i + 1  # Offset due to central difference
        A = A_history[snapshot_idx]
        B = B_history[snapshot_idx]
        t = times[snapshot_idx]

        # Spatial derivatives for both fields
        A_x, A_y, A_xx, A_yy = spatial_derivatives(A, dx, dy)
        B_x, B_y, B_xx, B_yy = spatial_derivatives(B, dx, dy)

        # Flatten spatial grids and append
        # Each (x, y) point at this timestep becomes a training sample
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
        training_data['u_t'].append(A_t[i].flatten())
        training_data['v_t'].append(B_t[i].flatten())

        # Coordinates and time (broadcast to all spatial points)
        X, Y = np.meshgrid(x, y)
        training_data['x_coord'].append(X.flatten())
        training_data['y_coord'].append(Y.flatten())
        training_data['t'].append(np.full(X.size, t))

    # Stack all timesteps together
    for key in training_data:
        training_data[key] = np.concatenate(training_data[key])

    n_samples = len(training_data['u'])
    print(f"  Generated {n_samples:,} training samples")
    print(f"  ({len(A_t)} timesteps × {A.size} spatial points)")

    return training_data


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


def main():
    """Main data generation workflow."""
    parser = argparse.ArgumentParser(description='Generate Brusselator training data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
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

    # Track success/failure
    successful = 0
    failed_tasks = []
    skipped_existing = 0

    # Process each IC
    for ic_idx, ic_config in enumerate(ic_configs):
        ic_name = ic_config.get('name', f'ic_{ic_idx}')

        # Check if already generated
        output_file = data_dir / f"{ic_name}.npz"
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
        for param in ['D_A', 'D_B', 'k1', 'k2']:
            raw_val = ic_config.get(param, task_sim_params[param])
            if isinstance(raw_val, list):
                if len(raw_val) != 2:
                    raise ValueError(
                        f"Task '{ic_name}': {param} as list must have exactly 2 elements [min, max], "
                        f"got {len(raw_val)} elements: {raw_val}"
                    )
                rng = np.random.default_rng(ic_config.get('seed'))
                task_sim_params[param] = rng.uniform(raw_val[0], raw_val[1])
                print(f"Sampled {param} = {task_sim_params[param]:.6f} from range {raw_val}")
            elif raw_val != task_sim_params[param]:
                task_sim_params[param] = raw_val

        # Inject k1, k2 into IC config for steady state calculation
        ic_config_with_params = ic_config.copy()
        ic_config_with_params['k1'] = task_sim_params['k1']
        ic_config_with_params['k2'] = task_sim_params['k2']

        try:
            # Create initial condition
            A_init, B_init, generated_params = create_brusselator_ic(ic_config_with_params, x, y)

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

            # Compute derivatives and format as training data
            training_data = generate_training_samples(concentration_history, times, x_result, y_result)

            # Validate data for silent divergence
            max_magnitude = 1e6
            diverged_silently = False
            for key in ['u', 'v', 'u_t', 'v_t']:
                if key in training_data:
                    max_val = np.abs(training_data[key]).max()
                    if max_val > max_magnitude or np.any(np.isnan(training_data[key])) or np.any(np.isinf(training_data[key])):
                        diverged_silently = True
                        print(f"\n⚠️  Silent divergence detected: {key} has max magnitude {max_val:.2e}")
                        break

            if diverged_silently:
                raise ValueError(f"Silent divergence: max values exceed {max_magnitude:.0e}")

            # Save data
            ic_config_to_save = ic_config.copy()
            ic_config_to_save['k1_used'] = task_sim_params['k1']
            ic_config_to_save['k2_used'] = task_sim_params['k2']
            ic_config_to_save['D_A_used'] = task_sim_params['D_A']
            ic_config_to_save['D_B_used'] = task_sim_params['D_B']

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

            # Generate visualization
            vis_file = data_dir / f"{ic_name}_evolution.png"
            save_brusselator_evolution(concentration_history, times, x_result, y_result, str(vis_file), n_snapshots=4)

            # Print sample data
            print(f"\nSample data point (first sample):")
            print(f"  A={training_data['u'][0]:.4f}, B={training_data['v'][0]:.4f}")
            print(f"  A_xx={training_data['u_xx'][0]:.4f}, A_yy={training_data['u_yy'][0]:.4f}")
            print(f"  A_t={training_data['u_t'][0]:.4f}, B_t={training_data['v_t'][0]:.4f}")
            print(f"  t={training_data['t'][0]:.4f}, x={training_data['x_coord'][0]:.4f}, y={training_data['y_coord'][0]:.4f}")

            successful += 1

        except Exception as e:
            print(f"\n⚠️  SKIPPED: {ic_name}")
            print(f"    Error: {type(e).__name__}: {str(e)}")
            failed_tasks.append((ic_name, "error", str(e)))

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
        for name, reason, details in failed_tasks:
            print(f"    - {name}: {reason} - {details}")

    if successful > 0:
        print(f"\nEach successful dataset contains:")
        print(f"  - Features: u(A), v(B), derivatives (10 inputs, NS-compatible format)")
        print(f"  - Targets: u_t(A_t), v_t(B_t) (2 outputs)")
        print(f"  - Metadata: t, x, y coordinates + .txt file with parameters")


if __name__ == '__main__':
    main()
