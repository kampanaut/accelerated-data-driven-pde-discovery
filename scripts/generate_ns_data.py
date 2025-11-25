"""
Generate Navier-Stokes training data for PDE discovery.

This script:
1. Defines initial conditions (Gaussian vortex configurations)
2. Solves Navier-Stokes equations for each IC
3. Computes spatial and temporal derivatives
4. Formats data as training samples for the N network
5. Saves data and visualizations
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.navier_stokes import solve_navier_stokes_with_params
from src.data.derivatives import spatial_derivatives, temporal_derivative
from src.utils.visualization import save_flow_evolution


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


def main():
    """Main data generation workflow."""
    print("=" * 60)
    print("Navier-Stokes Data Generation for PDE Discovery")
    print("=" * 60)

    # Simulation parameters (shared across all ICs)
    simulation_params = {
        'nu': 0.01,                     # Viscosity
        'domain_size': (2 * np.pi, 2 * np.pi),  # [0, 2π] × [0, 2π]
        'resolution': (32, 64),          # 32×64 grid (as agreed with supervisor)
        't_end': 10.0,                   # Simulate to t=10 (longer to see decay)
        'dt': 0.01,                      # Small timestep for stability
        'save_interval': 0.2,            # Save every 0.2 time units (fewer snapshots)
    }

    print(f"\nSimulation parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")

    # Define 3 initial conditions (Gaussian vortices at different locations)
    ic_configs = [
        {
            'name': 'vortex_center',
            'type': 'gaussian_vortex',
            'center': (np.pi, np.pi),
            'width': 0.5,
            'strength': 1.0,
        },
        {
            'name': 'vortex_left',
            'type': 'gaussian_vortex',
            'center': (np.pi / 2, np.pi),
            'width': 0.5,
            'strength': 1.5,
        },
        {
            'name': 'vortex_bottom',
            'type': 'gaussian_vortex',
            'center': (np.pi, np.pi / 2),
            'width': 0.5,
            'strength': 0.8,
        },
    ]

    print(f"\nInitial conditions: {len(ic_configs)} configurations")

    # Create output directory
    data_dir = Path(__file__).parent.parent / 'data' / 'navier_stokes'
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {data_dir}")

    # Process each IC
    for ic_idx, ic_params in enumerate(ic_configs):
        print(f"\n{'-' * 60}")
        print(f"IC {ic_idx + 1}/{len(ic_configs)}: {ic_params['name']}")
        print(f"{'-' * 60}")

        # Solve Navier-Stokes
        results = solve_navier_stokes_with_params(ic_params, simulation_params)

        # Extract results
        velocity_history = results['velocity_history']
        times = results['times']
        x = results['x']
        y = results['y']

        # Compute derivatives and format as training data
        training_data = generate_training_samples(velocity_history, times, x, y)

        # Save data
        output_file = data_dir / f"ic_{ic_idx:03d}.npz"
        np.savez(
            output_file,
            **training_data,
            ic_params=ic_params,
            simulation_params=simulation_params,
            x=x,
            y=y,
        )
        print(f"\nSaved training data to {output_file}")

        # Generate visualization
        vis_file = data_dir / f"ic_{ic_idx:03d}_evolution.png"
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        save_flow_evolution(velocity_history, times, x, y, dx, dy, str(vis_file), n_snapshots=4)

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
    print(f"\nNext steps:")
    print(f"  1. Inspect visualizations in {data_dir}")
    print(f"  2. Load data for N network training")
    print(f"  3. Implement MAML meta-learning loop")


if __name__ == '__main__':
    main()
