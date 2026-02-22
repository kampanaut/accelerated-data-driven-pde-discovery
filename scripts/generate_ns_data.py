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

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import argparse
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.navier_stokes import solve_navier_stokes_with_params
from src.utils.visualization import save_flow_evolution
from src.data import initial_conditions_ns


def generate_fourier_data(velocity_history: list, times: np.ndarray) -> dict:
    """
    Convert velocity snapshots to Fourier coefficients with temporal derivatives.

    FFTs each snapshot and computes temporal central differences on the
    coefficients. Drops first/last snapshot (same as grid pipeline).

    Unlike Brusselator Fourier data (which stores ALL snapshots and computes
    targets from PDE RHS), NS must pre-compute temporal derivatives because
    the pressure term prevents analytical RHS evaluation.

    Args:
        velocity_history: List of (u, v) tuples at different times
        times: Time values for each snapshot

    Returns:
        Dict with keys: u_hat, v_hat, u_t_hat, v_t_hat (complex128),
        times (float64), dt_save (float64)
    """
    n_snapshots = len(velocity_history)
    ny, nx = velocity_history[0][0].shape
    dt_save = float(times[1] - times[0])

    # FFT all snapshots
    u_hat_all = np.empty((n_snapshots, ny, nx), dtype=np.complex128)
    v_hat_all = np.empty((n_snapshots, ny, nx), dtype=np.complex128)

    for i, (u, v) in enumerate(velocity_history):
        u_hat_all[i] = np.fft.fft2(u)
        v_hat_all[i] = np.fft.fft2(v)

    # Temporal central difference on coefficients (drops first/last)
    # u_t_hat[i] ≈ (u_hat[i+1] - u_hat[i-1]) / (2 * dt_save)
    n_valid = n_snapshots - 2
    u_t_hat = (u_hat_all[2:] - u_hat_all[:-2]) / (2 * dt_save)
    v_t_hat = (v_hat_all[2:] - v_hat_all[:-2]) / (2 * dt_save)

    # Keep only interior snapshots (matching temporal derivative indices)
    u_hat = u_hat_all[1:-1]
    v_hat = v_hat_all[1:-1]
    valid_times = times[1:-1]

    print(
        f"  FFT'd {n_snapshots} snapshots → {n_valid} valid (central diff), shape ({ny}, {nx})"
    )

    return {
        "u_hat": u_hat,
        "v_hat": v_hat,
        "u_t_hat": u_t_hat,
        "v_t_hat": v_t_hat,
        "times": valid_times,
        "dt_save": np.float64(dt_save),
    }


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


def process_single_ic(args_tuple):
    """
    Worker function for multiprocessing Pool.

    Processes a single IC configuration, including retry logic for divergence.

    Args:
        args_tuple: (ic_config, simulation_params, data_dir, ic_idx, x, y, fourier_mode)

    Returns:
        (status, ic_name, error_msg, retry_count)
        status: 'success', 'skipped', or 'failed'
    """
    ic_config, simulation_params, data_dir, ic_idx, x, y = args_tuple

    ic_name = ic_config.get("name", f"ic_{ic_idx}")

    # Check if already generated
    output_file = Path(data_dir) / f"{ic_name}_fourier.npz"
    if output_file.exists():
        return ("skipped", ic_name, None, 0)

    # Check for task-specific viscosity override
    raw_nu = ic_config.get("nu", simulation_params["nu"])

    if isinstance(raw_nu, list) and len(raw_nu) != 2:
        return (
            "failed",
            ic_name,
            f"nu as list must have 2 elements, got {len(raw_nu)}",
            0,
        )

    # Retry loop for divergence
    max_retries = 800
    base_seed = ic_config.get("seed", None)

    for attempt in range(max_retries):
        rng = np.random.default_rng(
            base_seed + attempt * 1000 if base_seed is not None else None
        )

        task_sim_params = simulation_params.copy()
        task_nu = (
            rng.uniform(raw_nu[0], raw_nu[1]) if isinstance(raw_nu, list) else raw_nu
        )
        task_sim_params["nu"] = task_nu

        ic_config_attempt = ic_config.copy()
        if base_seed is not None:
            ic_config_attempt["seed"] = base_seed + attempt * 1000

        try:
            # Create initial condition
            u_init, v_init, _ = (
                create_ic_from_config(  # u_init, v_init, generated_params
                    ic_config_attempt, x, y
                )
            )

            # Wrap IC for solver
            ic_params_for_solver = {
                "type": "custom",
                "u_init": u_init,
                "v_init": v_init,
            }

            # Solve Navier-Stokes
            results = solve_navier_stokes_with_params(
                ic_params_for_solver, task_sim_params, task_name=ic_name
            )

            velocity_history = results["velocity_history"]
            times = results["times"]
            x_result = results["x"]
            y_result = results["y"]

            ic_config_to_save = ic_config.copy()
            ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
            ic_config_to_save["retry_attempt"] = attempt

            fourier_data = generate_fourier_data(velocity_history, times)

            # Validate Fourier data
            for key in ["u_hat", "v_hat", "u_t_hat", "v_t_hat"]:
                arr = fourier_data[key]
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise ValueError(f"Silent divergence: {key} has NaN/Inf")

            np.savez(
                output_file,
                **fourier_data,
                ic_config=ic_config_to_save,
                simulation_params=task_sim_params,
                nu_used=task_nu,
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

        except RuntimeError:
            print(f"{ic_name} Diverged")
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
            output_file = data_dir / f"{ic_name}_fourier.npz"
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

            if isinstance(raw_nu, list) and len(raw_nu) != 2:
                raise ValueError(
                    f"Task '{ic_name}': nu as list must have exactly 2 elements [min, max], "
                    f"got {len(raw_nu)} elements: {raw_nu}"
                )

            # Retry loop for divergence
            max_retries = 800
            base_seed = ic_config.get("seed", None)

            for attempt in range(max_retries):
                rng = np.random.default_rng(
                    base_seed + attempt * 1000 if base_seed is not None else None
                )

                task_sim_params = simulation_params.copy()
                task_nu = (
                    rng.uniform(raw_nu[0], raw_nu[1])
                    if isinstance(raw_nu, list)
                    else raw_nu
                )
                task_sim_params["nu"] = task_nu
                print(
                    f"{'Sampled' if isinstance(raw_nu, list) else 'Using'} ν = {task_nu:.6f}"
                )

                ic_config_attempt = ic_config.copy()
                if base_seed is not None:
                    ic_config_attempt["seed"] = base_seed + attempt * 1000
                    if attempt > 0:
                        print(
                            f"  Retry {attempt}/{max_retries - 1} with seed {ic_config_attempt['seed']}"
                        )

                try:
                    # Create initial condition
                    u_init, v_init, _ = (
                        create_ic_from_config(  # u_init, v_init, generated_params
                            ic_config_attempt, x, y
                        )
                    )

                    ic_params_for_solver = {
                        "type": "custom",
                        "u_init": u_init,
                        "v_init": v_init,
                    }

                    # Solve Navier-Stokes
                    results = solve_navier_stokes_with_params(
                        ic_params_for_solver, task_sim_params, task_name=ic_name
                    )

                    velocity_history = results["velocity_history"]
                    times = results["times"]
                    x_result = results["x"]
                    y_result = results["y"]

                    ic_config_to_save = ic_config.copy()
                    ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
                    ic_config_to_save["retry_attempt"] = attempt

                    fourier_data = generate_fourier_data(velocity_history, times)

                    # Validate Fourier data
                    for key in ["u_hat", "v_hat", "u_t_hat", "v_t_hat"]:
                        arr = fourier_data[key]
                        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                            raise ValueError(f"Silent divergence: {key} has NaN/Inf")

                    np.savez(
                        output_file,
                        **fourier_data,
                        ic_config=ic_config_to_save,
                        simulation_params=task_sim_params,  # type: ignore[reportArgumentType]
                        nu_used=task_nu,
                    )

                    print(f"\nSaved Fourier data to {output_file}")
                    if attempt > 0:
                        print(
                            f"  (succeeded after {attempt} retries, final seed: {ic_config_attempt.get('seed')})"
                        )

                    # Generate visualization (both modes)
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

                    successful += 1
                    break  # Success, exit retry loop

                except RuntimeError as e:
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
        print("  - Fourier coefficients: u_hat, v_hat, u_t_hat, v_t_hat (complex128)")
        print("  - Metadata: ic_config, simulation_params")


if __name__ == "__main__":
    main()
