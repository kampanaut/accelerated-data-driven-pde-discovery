"""
Generate Navier-Stokes training data for PDE discovery.

This script:
1. Loads initial condition configuration from YAML file
2. Solves Navier-Stokes equations for each IC
3. Computes spatial and temporal derivatives
4. Formats data as training samples for the N network
5. Saves data and visualizations

Usage:
    python scripts/generate_ns_data.py --config configs/gaussian_sweep.yaml
"""

import numpy as np
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.navier_stokes import solve_navier_stokes_with_params
from src.data.derivatives import spatial_derivatives, temporal_derivative
from src.utils.visualization import save_flow_evolution
from src.data import initial_conditions


def generate_training_samples(
    velocity_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray
) -> dict:
    """
    Convert velocity snapshots to training samples with derivatives.

    Args:
        velocity_history: List of (u, v) tuples at different times
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

    # Separate u and v histories
    u_history = [vel[0] for vel in velocity_history]
    v_history = [vel[1] for vel in velocity_history]

    # Compute temporal derivatives (loses first and last timesteps)
    u_t = temporal_derivative(u_history, dt)
    v_t = temporal_derivative(v_history, dt)

    print(f"  Temporal derivatives shape: {u_t.shape}")

    # Compute spatial derivatives for each snapshot (excluding first/last)
    training_data = {
        'u': [],
        'v': [],
        'u_x': [],
        'u_y': [],
        'u_xx': [],
        'u_yy': [],
        'v_x': [],
        'v_y': [],
        'v_xx': [],
        'v_yy': [],
        'u_t': [],
        'v_t': [],
        't': [],
        'x_coord': [],
        'y_coord': [],
    }

    # Process snapshots with valid time derivatives (indices 1 to n-2)
    for i in range(len(u_t)):
        snapshot_idx = i + 1  # Offset due to central difference
        u = u_history[snapshot_idx]
        v = v_history[snapshot_idx]
        t = times[snapshot_idx]

        # Spatial derivatives
        u_x, u_y, u_xx, u_yy = spatial_derivatives(u, dx, dy)
        v_x, v_y, v_xx, v_yy = spatial_derivatives(v, dx, dy)

        # Flatten spatial grids and append
        # Each (x, y) point at this timestep becomes a training sample
        training_data['u'].append(u.flatten())
        training_data['v'].append(v.flatten())
        training_data['u_x'].append(u_x.flatten())
        training_data['u_y'].append(u_y.flatten())
        training_data['u_xx'].append(u_xx.flatten())
        training_data['u_yy'].append(u_yy.flatten())
        training_data['v_x'].append(v_x.flatten())
        training_data['v_y'].append(v_y.flatten())
        training_data['v_xx'].append(v_xx.flatten())
        training_data['v_yy'].append(v_yy.flatten())
        training_data['u_t'].append(u_t[i].flatten())
        training_data['v_t'].append(v_t[i].flatten())

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
    print(f"  ({len(u_t)} timesteps × {u.size} spatial points)")

    return training_data


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_ic_from_config(ic_config: dict, x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Create initial condition from configuration dict.

    Args:
        ic_config: Dict with 'type' and type-specific parameters
        x, y: Coordinate arrays

    Returns:
        (u, v): Initial velocity field
    """
    ic_type = ic_config['type']

    if ic_type == 'gaussian_vortex':
        return initial_conditions.gaussian_vortex_ic(
            center=tuple(ic_config['center']),
            width=ic_config['width'],
            strength=ic_config['strength'],
            x=x,
            y=y
        )

    elif ic_type == 'multi_vortex':
        vortex_params = []
        for v in ic_config['vortices']:
            vortex_params.append({
                'center': tuple(v['center']),
                'width': v['width'],
                'strength': v['strength']
            })
        return initial_conditions.multi_vortex_ic(vortex_params, x, y)

    elif ic_type == 'taylor_green':
        amplitude = ic_config.get('amplitude', 1.0)
        return initial_conditions.taylor_green_vortex(x, y, amplitude)

    elif ic_type == 'shear_layer':
        return initial_conditions.shear_layer_ic(
            y_center=ic_config['y_center'],
            thickness=ic_config['thickness'],
            velocity_jump=ic_config['velocity_jump'],
            perturbation_amplitude=ic_config['perturbation_amplitude'],
            x=x,
            y=y
        )

    elif ic_type == 'lamb_oseen':
        return initial_conditions.lamb_oseen_vortex_ic(
            center=tuple(ic_config['center']),
            core_radius=ic_config['core_radius'],
            circulation=ic_config['circulation'],
            x=x,
            y=y
        )

    elif ic_type == 'dipole':
        return initial_conditions.dipole_vortex_ic(
            center=tuple(ic_config['center']),
            separation=ic_config['separation'],
            width=ic_config['width'],
            strength=ic_config['strength'],
            x=x,
            y=y
        )

    elif ic_type == 'perturbed_flow':
        return initial_conditions.perturbed_uniform_flow_ic(
            u_mean=ic_config['u_mean'],
            v_mean=ic_config['v_mean'],
            perturbation_amplitude=ic_config['perturbation_amplitude'],
            perturbation_wavelength=ic_config['perturbation_wavelength'],
            x=x,
            y=y
        )

    elif ic_type == 'random_soup':
        return initial_conditions.random_vortex_soup_ic(
            n_vortices=ic_config['n_vortices'],
            strength_range=tuple(ic_config['strength_range']),
            width_range=tuple(ic_config['width_range']),
            x=x,
            y=y,
            seed=ic_config.get('seed', None)
        )

    elif ic_type == 'von_karman':
        return initial_conditions.von_karman_street_ic(
            n_vortices=ic_config['n_vortices'],
            spacing=ic_config['spacing'],
            offset=ic_config['offset'],
            width=ic_config['width'],
            strength=ic_config['strength'],
            x=x,
            y=y
        )

    else:
        raise ValueError(f"Unknown IC type: {ic_type}")


def main():
    """Main data generation workflow."""
    parser = argparse.ArgumentParser(description='Generate Navier-Stokes training data')
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
    print("Navier-Stokes Data Generation for PDE Discovery")
    print("=" * 60)
    print(f"Config file: {config_path}")

    # Extract simulation parameters
    sim_params = config['simulation']
    simulation_params = {
        'nu': sim_params['nu'],
        'domain_size': tuple(sim_params['domain_size']),
        'resolution': tuple(sim_params['resolution']),
        't_end': sim_params['t_end'],
        'dt': sim_params['dt'],
        'save_interval': sim_params['save_interval'],
    }

    print(f"\nSimulation parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")

    # Extract IC configurations
    ic_configs = config['initial_conditions']
    print(f"\nInitial conditions: {len(ic_configs)} configurations")

    # Create output directory
    output_dir_name = config.get('output_dir', 'navier_stokes')
    data_dir = Path(__file__).parent.parent / 'data' / output_dir_name
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {data_dir}")

    # Setup coordinate arrays (needed for IC generation)
    domain_size = simulation_params['domain_size']
    resolution = simulation_params['resolution']
    x = np.linspace(0, domain_size[0], resolution[1])
    y = np.linspace(0, domain_size[1], resolution[0])

    # Process each IC
    for ic_idx, ic_config in enumerate(ic_configs):
        ic_name = ic_config.get('name', f'ic_{ic_idx}')
        print(f"\n{'-' * 60}")
        print(f"IC {ic_idx + 1}/{len(ic_configs)}: {ic_name} ({ic_config['type']})")
        print(f"{'-' * 60}")

        # Create initial condition
        u_init, v_init = create_ic_from_config(ic_config, x, y)

        # Wrap IC for solver
        ic_params_for_solver = {
            'type': 'custom',
            'u_init': u_init,
            'v_init': v_init,
        }

        # Solve Navier-Stokes
        results = solve_navier_stokes_with_params(ic_params_for_solver, simulation_params)

        # Extract results
        velocity_history = results['velocity_history']
        times = results['times']
        x_result = results['x']
        y_result = results['y']

        # Compute derivatives and format as training data
        training_data = generate_training_samples(velocity_history, times, x_result, y_result)

        # Save data
        output_file = data_dir / f"{ic_name}.npz"
        np.savez(
            output_file,
            **training_data,
            ic_config=ic_config,
            simulation_params=simulation_params,
            x=x_result,
            y=y_result,
        )
        print(f"\nSaved training data to {output_file}")

        # Generate visualization
        vis_file = data_dir / f"{ic_name}_evolution.png"
        dx = x_result[1] - x_result[0]
        dy = y_result[1] - y_result[0]
        save_flow_evolution(velocity_history, times, x_result, y_result, dx, dy, str(vis_file), n_snapshots=4)

        # Print sample data
        print(f"\nSample data point (first sample):")
        print(f"  u={training_data['u'][0]:.4f}, v={training_data['v'][0]:.4f}")
        print(f"  u_x={training_data['u_x'][0]:.4f}, u_y={training_data['u_y'][0]:.4f}")
        print(f"  u_t={training_data['u_t'][0]:.4f}, v_t={training_data['v_t'][0]:.4f}")
        print(f"  t={training_data['t'][0]:.4f}, x={training_data['x_coord'][0]:.4f}, y={training_data['y_coord'][0]:.4f}")

    print(f"\n{'=' * 60}")
    print("Data generation complete!")
    print(f"{'=' * 60}")
    print(f"\nGenerated {len(ic_configs)} datasets in {data_dir}")
    print(f"Each dataset contains:")
    print(f"  - Training samples: {len(training_data['u']):,} samples")
    print(f"  - Features: u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy")
    print(f"  - Targets: u_t, v_t")
    print(f"  - Metadata: t, x, y coordinates")


if __name__ == '__main__':
    main()
