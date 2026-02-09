"""
Generate Navier-Stokes training data for PDE discovery.

This script:
1. Loads initial condition configuration from YAML file
2. Solves Navier-Stokes equations for each IC
3. Computes spatial and temporal derivatives
4. Formats data as training samples for the N network
5. Saves data and visualizations

Usage:
    python scripts/generate_ns_data.py --config configs/ic_train.yaml
    python scripts/generate_ns_data.py --config configs/ic_train.yaml --gpu --workers 8
"""

import sys
import os
from pathlib import Path
from typing import Any, TypedDict

from numpy.typing import NDArray

# Default to CPU for stability (PhiFlow iterative solvers diverge more on GPU)
# Use --gpu to opt into GPU execution
if "--gpu" in sys.argv:
    sys.argv.remove("--gpu")
    # Clear JAX_PLATFORMS to let JAX auto-detect GPU
    os.environ.pop("JAX_PLATFORMS", None)
else:
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import argparse
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.navier_stokes import solve_navier_stokes_with_params
from src.data.derivatives import spatial_derivatives, temporal_derivative
from src.utils.visualization import save_flow_evolution
from src.data import initial_conditions_ns

# Import for catching divergence errors
try:
    from phiml.math._optimize import Diverged
except ImportError:
    # Fallback if phiml structure changes
    Diverged = Exception


class DatasetGridFormat(TypedDict):
    u: NDArray[np.floating[Any]]
    v: NDArray[np.floating[Any]]
    u_x: NDArray[np.floating[Any]]
    u_y: NDArray[np.floating[Any]]
    u_xx: NDArray[np.floating[Any]]
    u_yy: NDArray[np.floating[Any]]
    v_x: NDArray[np.floating[Any]]
    v_y: NDArray[np.floating[Any]]
    v_xx: NDArray[np.floating[Any]]
    v_yy: NDArray[np.floating[Any]]
    u_t: NDArray[np.floating[Any]]
    v_t: NDArray[np.floating[Any]]
    x_coord: NDArray[np.floating[Any]]
    y_coord: NDArray[np.floating[Any]]
    t: NDArray[np.floating[Any]]


def generate_training_samples(
    velocity_history: list, times: np.ndarray, x: np.ndarray, y: np.ndarray
) -> DatasetGridFormat:
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

    print("Computing derivatives...")
    print(f"  Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    print(f"  Time spacing: dt={dt:.4f}")

    # Separate u and v histories
    u_history = [vel[0] for vel in velocity_history]
    v_history = [vel[1] for vel in velocity_history]

    # Compute temporal derivatives (loses first and last timesteps)
    u_t = temporal_derivative(u_history, dt)
    v_t = temporal_derivative(v_history, dt)

    print(f"  Temporal derivatives shape: {u_t.shape}")

    # Compute spatial derivatives for each snapshot (excluding first/last due to central diff)
    n_valid = len(velocity_history) - 2
    column_size = n_valid * velocity_history[0][0].size
    training_data: DatasetGridFormat = {
        "u": np.zeros(column_size),
        "v": np.zeros(column_size),
        "u_x": np.zeros(column_size),
        "u_y": np.zeros(column_size),
        "u_xx": np.zeros(column_size),
        "u_yy": np.zeros(column_size),
        "v_x": np.zeros(column_size),
        "v_y": np.zeros(column_size),
        "v_xx": np.zeros(column_size),
        "v_yy": np.zeros(column_size),
        "u_t": np.zeros(column_size),
        "v_t": np.zeros(column_size),
        "t": np.zeros(column_size),
        "x_coord": np.zeros(column_size),
        "y_coord": np.zeros(column_size),
    }

    # Process snapshots with valid time derivatives (indices 1 to n-2)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()

    for i, (u, v) in enumerate(velocity_history[1:-1]):
        t = times[i + 1]  # Offset due to central difference

        # Spatial derivatives
        u_x, u_y, u_xx, u_yy = spatial_derivatives(u, dx, dy)
        v_x, v_y, v_xx, v_yy = spatial_derivatives(v, dx, dy)

        # Flatten and write into pre-allocated arrays
        start = u.size * i
        end = start + u.size
        training_data["u"][start:end] = u.flatten()
        training_data["v"][start:end] = v.flatten()
        training_data["u_x"][start:end] = u_x.flatten()
        training_data["u_y"][start:end] = u_y.flatten()
        training_data["u_xx"][start:end] = u_xx.flatten()
        training_data["u_yy"][start:end] = u_yy.flatten()
        training_data["v_x"][start:end] = v_x.flatten()
        training_data["v_y"][start:end] = v_y.flatten()
        training_data["v_xx"][start:end] = v_xx.flatten()
        training_data["v_yy"][start:end] = v_yy.flatten()
        training_data["u_t"][start:end] = u_t[i].flatten()
        training_data["v_t"][start:end] = v_t[i].flatten()
        training_data["x_coord"][start:end] = x_flat
        training_data["y_coord"][start:end] = y_flat
        training_data["t"][start:end] = np.full(X.size, t)

    n_samples = len(training_data["u"])
    print(f"  Generated {n_samples:,} training samples")
    print(f"  ({n_valid} timesteps × {velocity_history[0][0].size} spatial points)")

    return training_data


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_ic_from_config(ic_config: dict, x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Create initial condition from configuration dict.

    Args:
        ic_config: Dict with 'type' and type-specific parameters
        x, y: Coordinate arrays

    Returns:
        (u, v, generated_params): Initial velocity field and generated parameters
    """
    ic_type = ic_config["type"]

    if ic_type == "gaussian_hill":
        return initial_conditions_ns.gaussian_hill_ic(
            center=tuple(ic_config["center"]),
            width=ic_config["width"],
            strength=ic_config["strength"],
            x=x,
            y=y,
        )

    elif ic_type == "multi_vortex":
        vortex_params = []
        for v in ic_config["vortices"]:
            vortex_params.append(
                {
                    "center": tuple(v["center"]),
                    "width": v["width"],
                    "strength": v["strength"],
                }
            )
        return initial_conditions_ns.multi_vortex_ic(vortex_params, x, y)

    elif ic_type == "taylor_green":
        amplitude = ic_config.get("amplitude", 1.0)
        return initial_conditions_ns.taylor_green_vortex(x, y, amplitude)

    elif ic_type == "shear_layer":
        return initial_conditions_ns.shear_layer_ic(
            y_center=ic_config["y_center"],
            thickness=ic_config["thickness"],
            velocity_jump=ic_config["velocity_jump"],
            perturbation_amplitude=ic_config["perturbation_amplitude"],
            x=x,
            y=y,
        )

    elif ic_type == "lamb_oseen":
        return initial_conditions_ns.lamb_oseen_vortex_ic(
            center=tuple(ic_config["center"]),
            core_radius=ic_config["core_radius"],
            circulation=ic_config["circulation"],
            x=x,
            y=y,
        )

    elif ic_type == "dipole":
        return initial_conditions_ns.dipole_vortex_ic(
            center=tuple(ic_config["center"]),
            separation=ic_config["separation"],
            width=ic_config["width"],
            strength=ic_config["strength"],
            x=x,
            y=y,
        )

    elif ic_type == "perturbed_flow":
        return initial_conditions_ns.perturbed_uniform_flow_ic(
            u_mean=ic_config["u_mean"],
            v_mean=ic_config["v_mean"],
            perturbation_amplitude=ic_config["perturbation_amplitude"],
            perturbation_wavelength=ic_config["perturbation_wavelength"],
            x=x,
            y=y,
            seed=ic_config["seed"],
        )

    elif ic_type == "random_soup":
        return initial_conditions_ns.random_vortex_soup_ic(
            n_vortices=ic_config["n_vortices"],
            strength_range=tuple(ic_config["strength_range"]),
            width_range=tuple(ic_config["width_range"]),
            x=x,
            y=y,
            seed=ic_config["seed"],
        )

    elif ic_type == "von_karman":
        return initial_conditions_ns.von_karman_street_ic(
            n_vortices=ic_config["n_vortices"],
            spacing=ic_config["spacing"],
            offset=ic_config["offset"],
            width=ic_config["width"],
            strength=ic_config["strength"],
            x=x,
            y=y,
        )

    elif ic_type == "gaussian_vortex":
        return initial_conditions_ns.gaussian_vortex_ic(
            n_gaussians=ic_config["n_gaussians"],
            amplitude_range=tuple(ic_config["amplitude_range"]),
            width_range=tuple(ic_config["width_range"]),
            x=x,
            y=y,
            seed=ic_config["seed"],
        )

    elif ic_type == "gaussian_direct":
        return initial_conditions_ns.gaussian_direct_ic(
            n_gaussians_u=ic_config["n_gaussians_u"],
            n_gaussians_v=ic_config["n_gaussians_v"],
            amplitude_range=tuple(ic_config["amplitude_range"]),
            width_range=tuple(ic_config["width_range"]),
            x=x,
            y=y,
            seed=ic_config["seed"],
        )

    elif ic_type == "gaussian_hybrid":
        return initial_conditions_ns.gaussian_hybrid_ic(
            n_gaussians_vorticity=ic_config["n_gaussians_vorticity"],
            n_gaussians_u=ic_config["n_gaussians_u"],
            n_gaussians_v=ic_config["n_gaussians_v"],
            amplitude_range=tuple(ic_config["amplitude_range"]),
            width_range=tuple(ic_config["width_range"]),
            alpha=ic_config["alpha"],
            beta=ic_config["beta"],
            x=x,
            y=y,
            seed=ic_config["seed"],
        )

    else:
        raise ValueError(f"Unknown IC type: {ic_type}")


def save_metadata_txt(
    output_path: Path,
    ic_config: dict,
    simulation_params: dict,
    training_data: DatasetGridFormat,
    generated_params: dict,
    success: bool,
    error_msg: str,
):
    """
    Save human-readable metadata alongside .npz file.
    Shows exact parameters used for reproducibility/debugging.

    Args:
        output_path: Path to .npz file (will create .txt with same name)
        ic_config: Initial condition configuration
        simulation_params: Simulation parameters used
        training_data: Generated training data (for statistics)
        generated_params: Actual parameters generated by IC function (centers, widths, etc.)
        success: Whether simulation succeeded
        error_msg: Error message if failed
    """
    txt_path = output_path.with_suffix(".txt")

    with open(txt_path, "w") as f:
        f.write("Navier-Stokes Simulation Metadata\n")
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
            if key not in ["type", "name"]:
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Generated parameters (actual values used)
        if generated_params:
            f.write("Generated Parameters (actual values):\n")
            _write_generated_params(f, generated_params)
            f.write("\n")

        # Simulation parameters
        f.write("Simulation Parameters:\n")
        f.write(f"  ν (viscosity): {simulation_params['nu']:.6f}\n")
        f.write(f"  Domain: {simulation_params['domain_size']}\n")
        f.write(f"  Resolution: {simulation_params['resolution']}\n")
        f.write(f"  Time range: [0, {simulation_params['t_end']}]\n")
        f.write(f"  Time step: dt={simulation_params['dt']}\n")
        if "save_interval" in simulation_params:
            f.write(f"  Save interval: {simulation_params['save_interval']}\n")
        f.write("\n")

        if success and training_data:
            # Output statistics
            f.write("Output Statistics:\n")
            f.write(f"  Total samples: {len(training_data['u']):,}\n")
            f.write("  Features: 10 (u, v, spatial derivatives)\n")
            f.write("  Targets: 2 (u_t, v_t)\n")
            f.write("\n")

            # Data ranges
            f.write("Data Ranges:\n")
            for key in ["u", "v", "u_t", "v_t"]:
                if key in training_data:
                    values = training_data[key]
                    f.write(
                        f"  {key:4s}: [{values.min():>8.4f}, {values.max():>8.4f}]\n"
                    )
            f.write("\n")

            # Check for potential issues
            f.write("Data Quality Checks:\n")
            has_nan = any(
                np.any(np.isnan(training_data[k])) for k in ["u", "v", "u_t", "v_t"]
            )
            has_inf = any(
                np.any(np.isinf(training_data[k])) for k in ["u", "v", "u_t", "v_t"]
            )
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
                # Format as coordinates
                f.write(f"{prefix}{key}: ({value[0]:.4f}, {value[1]:.4f})\n")
            elif isinstance(value, float):
                f.write(f"{prefix}{key}: {value:.6f}\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")
    else:
        f.write(f"{prefix}{params}\n")


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
    ic_config, simulation_params, data_dir, ic_idx, x, y = args_tuple

    ic_name = ic_config.get("name", f"ic_{ic_idx}")

    # Check if already generated
    output_file = Path(data_dir) / f"{ic_name}.npz"
    if output_file.exists():
        return ("skipped", ic_name, None, 0)

    # Check for task-specific viscosity override
    raw_nu = ic_config.get("nu", simulation_params["nu"])

    if isinstance(raw_nu, list):
        if len(raw_nu) != 2:
            return (
                "failed",
                ic_name,
                f"nu as list must have 2 elements, got {len(raw_nu)}",
                0,
            )
        rng = np.random.default_rng(ic_config.get("seed"))
        task_nu = rng.uniform(raw_nu[0], raw_nu[1])
    else:
        task_nu = raw_nu

    task_sim_params = simulation_params.copy()
    task_sim_params["nu"] = task_nu

    # Retry loop for divergence
    max_retries = 800
    base_seed = ic_config.get("seed", None)

    for attempt in range(max_retries):
        ic_config_attempt = ic_config.copy()
        if base_seed is not None:
            ic_config_attempt["seed"] = base_seed + attempt * 1000

        try:
            # Create initial condition
            u_init, v_init, generated_params = create_ic_from_config(
                ic_config_attempt, x, y
            )

            # Wrap IC for solver
            ic_params_for_solver = {
                "type": "custom",
                "u_init": u_init,
                "v_init": v_init,
            }

            # Solve Navier-Stokes
            results = solve_navier_stokes_with_params(
                ic_params_for_solver, task_sim_params
            )

            velocity_history = results["velocity_history"]
            times = results["times"]
            x_result = results["x"]
            y_result = results["y"]

            # Compute derivatives and format as training data
            training_data = generate_training_samples(
                velocity_history, times, x_result, y_result
            )

            # Validate data for silent divergence
            max_magnitude = 1e6
            for key in ["u", "v", "u_t", "v_t"]:
                if key in training_data:
                    max_val = np.abs(training_data[key]).max()
                    if (
                        max_val > max_magnitude
                        or np.any(np.isnan(training_data[key]))
                        or np.any(np.isinf(training_data[key]))
                    ):
                        raise ValueError(
                            f"Silent divergence: {key} has max magnitude {max_val:.2e}"
                        )

            # Save data
            ic_config_to_save = ic_config.copy()
            ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
            ic_config_to_save["retry_attempt"] = attempt

            np.savez(
                output_file,
                **training_data,  # type: ignore[reportArgumentType]
                ic_config=ic_config_to_save,
                simulation_params=task_sim_params,
                nu_used=task_nu,
                x=x_result,
                y=y_result,
            )

            # Save metadata
            save_metadata_txt(
                output_file,
                ic_config_to_save,
                task_sim_params,
                training_data,
                generated_params,
                success=True,
                error_msg="No Errors",
            )

            # Generate visualization
            vis_file = Path(data_dir) / f"{ic_name}_evolution.png"
            dx = x_result[1] - x_result[0]
            dy = y_result[1] - y_result[0]
            save_flow_evolution(
                velocity_history,
                times,
                x_result,
                y_result,
                dx,
                dy,
                str(vis_file),
                n_snapshots=4,
            )

            return ("success", ic_name, None, attempt)

        except Diverged:
            if attempt < max_retries - 1:
                continue
            return (
                "failed",
                ic_name,
                f"Diverged after {max_retries} attempts",
                max_retries,
            )

        except ValueError as e:
            if "divergence" in str(e).lower() and attempt < max_retries - 1:
                continue
            if attempt >= max_retries - 1:
                return ("failed", ic_name, str(e), max_retries)
            continue

        except Exception as e:
            return ("failed", ic_name, f"{type(e).__name__}: {str(e)}", attempt)

    return ("failed", ic_name, f"Exhausted {max_retries} retries", max_retries)


def main():
    """Main data generation workflow."""
    parser = argparse.ArgumentParser(description="Generate Navier-Stokes training data")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = serial)",
    )
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
    sim_params = config["simulation"]
    simulation_params = {
        "nu": sim_params["nu"],
        "domain_size": tuple(sim_params["domain_size"]),
        "resolution": tuple(sim_params["resolution"]),
        "t_end": sim_params["t_end"],
        "dt": sim_params["dt"],
        "save_interval": sim_params["save_interval"],
    }

    print("\nSimulation parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")

    # Extract IC configurations
    ic_configs = config["initial_conditions"]
    print(f"\nInitial conditions: {len(ic_configs)} configurations")

    # Create output directory
    # Supports both old format (output_dir) and new format (simulation.name + output.base_dir)
    sim_name = config.get("simulation", {}).get("name") or config.get(
        "output_dir", "navier_stokes"
    )
    base_dir = config.get("output", {}).get("base_dir", "data/datasets")
    data_dir = Path(__file__).parent.parent / base_dir / sim_name
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {data_dir}")

    # Setup coordinate arrays (needed for IC generation)
    domain_size = simulation_params["domain_size"]
    resolution = simulation_params["resolution"]
    x = np.linspace(0, domain_size[0], resolution[1])
    y = np.linspace(0, domain_size[1], resolution[0])

    # Build work items for parallel processing
    work_items = [
        (ic_config, simulation_params, str(data_dir), ic_idx, x, y)
        for ic_idx, ic_config in enumerate(ic_configs)
    ]

    # Process ICs (parallel or serial)
    if args.workers > 1:
        # Parallel execution with compact progress output
        # Use 'spawn' context to avoid JAX fork() deadlock
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")

        print(f"\nUsing {args.workers} parallel workers")

        results = []
        with ctx.Pool(args.workers) as pool:
            for result in pool.imap_unordered(process_single_ic, work_items):
                status, name, error, retries = result
                results.append(result)

                # Progress reporting
                done = len(results)
                if status == "success":
                    retry_info = f" (after {retries} retries)" if retries > 0 else ""
                    print(f"[{done}/{len(ic_configs)}] {name}: ✓ SUCCESS{retry_info}")
                elif status == "skipped":
                    print(f"[{done}/{len(ic_configs)}] {name}: ⊘ SKIPPED (exists)")
                else:
                    print(f"[{done}/{len(ic_configs)}] {name}: ✗ FAILED - {error}")

        # Tally results
        successful = sum(1 for r in results if r[0] in ("success", "skipped"))
        skipped_existing = sum(1 for r in results if r[0] == "skipped")
        failed_tasks = [(r[1], r[2]) for r in results if r[0] == "failed"]

    else:
        # Serial execution with original verbose output
        successful = 0
        failed_tasks = []
        skipped_existing = 0

        for ic_idx, ic_config in enumerate(ic_configs):
            ic_name = ic_config.get("name", f"ic_{ic_idx}")

            # Check if already generated
            output_file = data_dir / f"{ic_name}.npz"
            if output_file.exists():
                print(
                    f"[{ic_idx + 1}/{len(ic_configs)}] {ic_name}: already exists, skipping"
                )
                skipped_existing += 1
                successful += 1
                continue

            print(f"\n{'-' * 60}")
            print(f"IC {ic_idx + 1}/{len(ic_configs)}: {ic_name} ({ic_config['type']})")
            print(f"{'-' * 60}")

            # Check for task-specific viscosity override
            raw_nu = ic_config.get("nu", simulation_params["nu"])

            if isinstance(raw_nu, list):
                if len(raw_nu) != 2:
                    raise ValueError(
                        f"Task '{ic_name}': nu as list must have exactly 2 elements [min, max], "
                        f"got {len(raw_nu)} elements: {raw_nu}"
                    )
                rng = np.random.default_rng(ic_config.get("seed"))
                task_nu = rng.uniform(raw_nu[0], raw_nu[1])
                print(f"Sampled ν = {task_nu:.6f} from range {raw_nu}")
            else:
                task_nu = raw_nu
                print(f"Using ν = {task_nu:.6f}")

            task_sim_params = simulation_params.copy()
            task_sim_params["nu"] = task_nu

            # Retry loop for divergence
            max_retries = 800
            base_seed = ic_config.get("seed", None)

            for attempt in range(max_retries):
                ic_config_attempt = ic_config.copy()
                if base_seed is not None:
                    ic_config_attempt["seed"] = base_seed + attempt * 1000
                    if attempt > 0:
                        print(
                            f"  Retry {attempt}/{max_retries - 1} with seed {ic_config_attempt['seed']}"
                        )

                try:
                    # Create initial condition
                    u_init, v_init, generated_params = create_ic_from_config(
                        ic_config_attempt, x, y
                    )

                    ic_params_for_solver = {
                        "type": "custom",
                        "u_init": u_init,
                        "v_init": v_init,
                    }

                    # Solve Navier-Stokes
                    results = solve_navier_stokes_with_params(
                        ic_params_for_solver, task_sim_params
                    )

                    velocity_history = results["velocity_history"]
                    times = results["times"]
                    x_result = results["x"]
                    y_result = results["y"]

                    # Compute derivatives and format as training data
                    training_data = generate_training_samples(
                        velocity_history, times, x_result, y_result
                    )

                    # Validate data for silent divergence
                    max_magnitude = 1e6
                    diverged_silently = False
                    for key in ["u", "v", "u_t", "v_t"]:
                        if key in training_data:
                            max_val = np.abs(training_data[key]).max()
                            if (
                                max_val > max_magnitude
                                or np.any(np.isnan(training_data[key]))
                                or np.any(np.isinf(training_data[key]))
                            ):
                                diverged_silently = True
                                print(
                                    f"\n⚠️  Silent divergence detected: {key} has max magnitude {max_val:.2e}"
                                )
                                break

                    if diverged_silently:
                        raise ValueError(
                            f"Silent divergence: max values exceed {max_magnitude:.0e}"
                        )

                    # Save data
                    ic_config_to_save = ic_config.copy()
                    ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
                    ic_config_to_save["retry_attempt"] = attempt

                    np.savez(
                        output_file,
                        **training_data,  # type: ignore[reportArgumentType]
                        ic_config=ic_config_to_save,
                        simulation_params=task_sim_params,  # type: ignore[reportArgumentType]
                        nu_used=task_nu,
                        x=x_result,
                        y=y_result,
                    )
                    print(f"\nSaved training data to {output_file}")
                    if attempt > 0:
                        print(
                            f"  (succeeded after {attempt} retries, final seed: {ic_config_attempt.get('seed')})"
                        )

                    # Save metadata
                    save_metadata_txt(
                        output_file,
                        ic_config_to_save,
                        task_sim_params,
                        training_data,
                        generated_params,
                        success=True,
                        error_msg="No Errors",
                    )

                    # Generate visualization
                    vis_file = data_dir / f"{ic_name}_evolution.png"
                    dx = x_result[1] - x_result[0]
                    dy = y_result[1] - y_result[0]
                    save_flow_evolution(
                        velocity_history,
                        times,
                        x_result,
                        y_result,
                        dx,
                        dy,
                        str(vis_file),
                        n_snapshots=4,
                    )

                    # Print sample data
                    print("\nSample data point (first sample):")
                    print(
                        f"  u={training_data['u'][0]:.4f}, v={training_data['v'][0]:.4f}"
                    )
                    print(
                        f"  u_x={training_data['u_x'][0]:.4f}, u_y={training_data['u_y'][0]:.4f}"
                    )
                    print(
                        f"  u_t={training_data['u_t'][0]:.4f}, v_t={training_data['v_t'][0]:.4f}"
                    )
                    print(
                        f"  t={training_data['t'][0]:.4f}, x={training_data['x_coord'][0]:.4f}, y={training_data['y_coord'][0]:.4f}"
                    )

                    successful += 1
                    break  # Success, exit retry loop

                except Diverged as e:
                    if attempt < max_retries - 1:
                        print("  ⚠️  Diverged, will retry...")
                        continue
                    print(f"\n⚠️  SKIPPED: {ic_name}")
                    print(
                        "    Reason: IC violated incompressibility too severely for pressure projection"
                    )
                    print(f"    Details: {str(e)}")
                    print(f"    (failed all {max_retries} attempts)")
                    failed_tasks.append((ic_name, str(e)))

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
                    print(f"\n⚠️  SKIPPED: {ic_name}")
                    print(f"    Unexpected error: {type(e).__name__}: {str(e)}")
                    failed_tasks.append((ic_name, str(e)))
                    break

    print(f"\n{'=' * 60}")
    print("Data generation complete!")
    print(f"{'=' * 60}")
    print("\nSummary:")
    print(f"  Total tasks: {len(ic_configs)}")
    print(f"  ✓ Successful: {successful} ({skipped_existing} already existed)")
    print(f"  ✗ Failed: {len(failed_tasks)}")
    print(f"\nOutput directory: {data_dir}")

    if failed_tasks:
        print("\n⚠️  Failed tasks:")
        for name, error in failed_tasks:
            print(f"    - {name}: {error}")

    if successful > 0:
        print("\nEach successful dataset contains:")
        print(
            "  - Features: u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy (10 inputs)"
        )
        print("  - Targets: u_t, v_t (2 outputs)")
        print("  - Metadata: t, x, y coordinates + .txt file with parameters")


if __name__ == "__main__":
    main()
