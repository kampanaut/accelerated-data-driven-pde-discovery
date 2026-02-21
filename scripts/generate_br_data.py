"""
Generate Brusselator training data for PDE discovery.

This script:
1. Loads initial condition configuration from YAML file
2. Solves Brusselator equations for each IC
3. Computes spatial and temporal derivatives
4. Formats data as training samples for the N network
5. Saves data and visualizations

Usage:
    python scripts/generate_br_data.py --config configs/brusselator_train.yaml
    python scripts/generate_br_data.py --config configs/brusselator_train.yaml --gpu --workers 8
"""

import sys
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import argparse
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.brusselator import solve_brusselator
from src.data.initial_conditions_brusselator import (
    create_brusselator_ic,
    compute_turing_threshold,
)


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
        concentration_history: List of (u, v) tuples at different times
        times: Time values for each snapshot

    Returns:
        Dict with keys: u_hat, v_hat (complex128), times (float64)
    """
    n_snapshots = len(concentration_history)
    ny, nx = concentration_history[0][0].shape

    u_hat_stack = np.empty((n_snapshots, ny, nx), dtype=np.complex128)
    v_hat_stack = np.empty((n_snapshots, ny, nx), dtype=np.complex128)

    for i, (u, v) in enumerate(concentration_history):
        u_hat_stack[i] = np.fft.fft2(u)
        v_hat_stack[i] = np.fft.fft2(v)

    print(f"  FFT'd {n_snapshots} snapshots, shape ({ny}, {nx})")

    return {
        "u_hat": u_hat_stack,
        "v_hat": v_hat_stack,
        "times": times,
    }


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_brusselator_evolution(
    concentration_history: list,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    n_snapshots: int = 4,
):
    """
    Save a multi-panel figure showing Brusselator evolution over time.

    Creates a 4×n_snapshots grid:
    - Row 1: A concentration (2D contour)
    - Row 2: B concentration (2D contour)
    - Row 3: A concentration (3D surface)
    - Row 4: B concentration (3D surface)

    Args:
        concentration_history: List of (u, v) tuples at different times
        times: Array of time values
        x, y: Coordinate arrays
        output_path: Path to save the figure
        n_snapshots: Number of snapshots to show
    """
    import matplotlib.pyplot as plt

    indices = np.linspace(0, len(concentration_history) - 1, n_snapshots, dtype=int)

    fig = plt.figure(figsize=(5 * n_snapshots, 20))

    # Prepare grid
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = np.meshgrid(x, y)
    else:
        x_grid, y_grid = x, y

    # Get global min/max for consistent colorbars
    all_u = [concentration_history[i][0] for i in indices]
    all_v = [concentration_history[i][1] for i in indices]
    u_min, u_max = min(a.min() for a in all_u), max(a.max() for a in all_u)
    v_min, v_max = min(b.min() for b in all_v), max(b.max() for b in all_v)

    for col, idx in enumerate(indices):
        u, v = concentration_history[idx]
        t = times[idx]

        # Row 1: u concentration (2D)
        ax_u_2d = plt.subplot(4, n_snapshots, col + 1)
        contour_u = ax_u_2d.contourf(
            x_grid, y_grid, u, levels=20, cmap="YlOrRd", vmin=u_min, vmax=u_max
        )
        plt.colorbar(contour_u, ax=ax_u_2d, fraction=0.046, pad=0.04)
        ax_u_2d.set_title(f"t={t:.3f}")
        ax_u_2d.set_aspect("equal")

        # Row 2: v concentration (2D)
        ax_v_2d = plt.subplot(4, n_snapshots, n_snapshots + col + 1)
        contour_v = ax_v_2d.contourf(
            x_grid, y_grid, v, levels=20, cmap="YlGnBu", vmin=v_min, vmax=v_max
        )
        plt.colorbar(contour_v, ax=ax_v_2d, fraction=0.046, pad=0.04)
        ax_v_2d.set_aspect("equal")

        # Row 3: u concentration (3D surface)
        ax_u_3d = plt.subplot(
            4, n_snapshots, 2 * n_snapshots + col + 1, projection="3d"
        )

        assert isinstance(ax_u_3d, Axes3D)

        surf_u = ax_u_3d.plot_surface(
            x_grid, y_grid, u, cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.9
        )

        ax_u_3d.set_xlabel("x")
        ax_u_3d.set_ylabel("y")
        ax_u_3d.set_zlabel("u")
        ax_u_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_u, ax=ax_u_3d, fraction=0.03, pad=0.1, shrink=0.5)

        # Row 4: v concentration (3D surface)
        ax_v_3d = plt.subplot(
            4, n_snapshots, 3 * n_snapshots + col + 1, projection="3d"
        )

        assert isinstance(ax_v_3d, Axes3D)

        surf_v = ax_v_3d.plot_surface(
            x_grid, y_grid, v, cmap="YlGnBu", linewidth=0, antialiased=True, alpha=0.9
        )
        ax_v_3d.set_xlabel("x")
        ax_v_3d.set_ylabel("y")
        ax_v_3d.set_zlabel("v")
        ax_v_3d.view_init(elev=30, azim=45)
        fig.colorbar(surf_v, ax=ax_v_3d, fraction=0.03, pad=0.1, shrink=0.5)

    # Add row labels
    fig.text(
        0.02,
        0.88,
        "u conc (2D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.02,
        0.65,
        "v conc (2D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.02,
        0.42,
        "u conc (3D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.02,
        0.18,
        "v conc (3D)",
        va="center",
        rotation="vertical",
        fontsize=11,
        weight="bold",
    )

    fig.suptitle("Brusselator Evolution", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
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
    ic_config, simulation_params, data_dir, ic_idx, x, y = args_tuple

    ic_name = ic_config.get("name", f"ic_{ic_idx}")

    # Check if already generated
    output_file = Path(data_dir) / f"{ic_name}_fourier.npz"
    if output_file.exists():
        return ("skipped", ic_name, None, 0)

    # Check for task-specific parameter overrides
    task_sim_params = simulation_params.copy()

    # Handle parameter sampling from ranges
    # Single RNG per task — draws are sequential so D_u, D_v, k1, k2 get distinct values
    param_rng = np.random.default_rng(ic_config.get("seed"))
    for param in ["D_u", "D_v", "k1", "k2"]:
        raw_val = ic_config.get(param, task_sim_params[param])
        if isinstance(raw_val, list):
            if len(raw_val) != 2:
                return (
                    "failed",
                    ic_name,
                    f"{param} as list must have 2 elements, got {len(raw_val)}",
                    0,
                )
            task_sim_params[param] = param_rng.uniform(raw_val[0], raw_val[1])
        elif raw_val != task_sim_params[param]:
            task_sim_params[param] = raw_val

    # k2_delta: sample k2 relative to Turing threshold instead of absolute range
    k2_delta_range = ic_config.get("k2_delta")
    k2_c = 0
    if k2_delta_range is not None:
        k2_c = compute_turing_threshold(
            k1=task_sim_params["k1"],
            D_u=task_sim_params["D_u"],
            D_v=task_sim_params["D_v"],
        )
        delta = param_rng.uniform(k2_delta_range[0], k2_delta_range[1])
        task_sim_params["k2"] = k2_c + delta

    # Inject k1, k2 into IC config for steady state calculation
    ic_config_with_params = ic_config.copy()
    ic_config_with_params["k1"] = task_sim_params["k1"]
    ic_config_with_params["k2"] = task_sim_params["k2"]

    # Retry loop for divergence
    max_retries = 800
    base_seed = ic_config.get("seed", None)

    for attempt in range(max_retries):
        ic_config_attempt = ic_config_with_params.copy()
        if base_seed is not None:
            ic_config_attempt["seed"] = base_seed + attempt * 1000

        try:
            # Create initial condition
            u_init, v_init, _ = create_brusselator_ic( # u_init, v_init, generated_params
                ic_config_attempt, x, y
            )

            # Solve Brusselator
            concentration_history, times, x_result, y_result = solve_brusselator(
                initial_concentration=(u_init, v_init),
                D_u=task_sim_params["D_u"],
                D_v=task_sim_params["D_v"],
                k1=task_sim_params["k1"],
                k2=task_sim_params["k2"],
                domain_size=task_sim_params["domain_size"],
                t_end=task_sim_params["t_end"],
                dt=task_sim_params["dt"],
                save_interval=task_sim_params.get("save_interval"),
            )

            # Validate raw concentrations for divergence
            max_magnitude = 1e6
            last_u, last_v = concentration_history[-1]
            for label, arr in [("u", last_u), ("v", last_v)]:
                max_val = np.abs(arr).max()
                if (
                    max_val > max_magnitude
                    or np.any(np.isnan(arr))
                    or np.any(np.isinf(arr))
                ):
                    raise ValueError(
                        f"Silent divergence: {label} has max magnitude {max_val:.2e}"
                    )

            ic_config_to_save = ic_config.copy()
            ic_config_to_save["k1_used"] = task_sim_params["k1"]
            ic_config_to_save["k2_used"] = task_sim_params["k2"]
            ic_config_to_save["D_u_used"] = task_sim_params["D_u"]
            ic_config_to_save["D_v_used"] = task_sim_params["D_v"]
            ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
            ic_config_to_save["retry_attempt"] = attempt
            if k2_delta_range is not None:
                assert k2_c > 0
                ic_config_to_save["k2_c"] = k2_c
                ic_config_to_save["k2_delta_used"] = task_sim_params["k2"] - k2_c

            fourier_data = generate_fourier_data(concentration_history, times)
            np.savez(
                output_file,
                **fourier_data,
                ic_config=ic_config_to_save,
                simulation_params=task_sim_params,
            )

            # Generate visualization
            vis_file = Path(data_dir) / f"{ic_name}_evolution.png"
            save_brusselator_evolution(
                concentration_history,
                times,
                x_result,
                y_result,
                str(vis_file),
                n_snapshots=4,
            )

            return ("success", ic_name, None, attempt)

        except ValueError as e:
            if "divergence" in str(e).lower() and attempt < max_retries - 1:
                continue
            if attempt >= max_retries - 1:
                return ("failed", ic_name, str(e), max_retries)
            continue

        except Exception as e:
            # Check if it's a divergence-related error
            error_str = str(e).lower()
            if (
                any(kw in error_str for kw in ["diverge", "nan", "inf", "overflow"])
                and attempt < max_retries - 1
            ):
                continue
            return ("failed", ic_name, f"{type(e).__name__}: {str(e)}", attempt)

    return ("failed", ic_name, f"Exhausted {max_retries} retries", max_retries)


def main():
    """Main data generation workflow."""
    parser = argparse.ArgumentParser(description="Generate Brusselator training data")
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
    print("Brusselator Data Generation for PDE Discovery")
    print("=" * 60)
    print(f"Config file: {config_path}")

    # Extract simulation parameters
    sim_params = config["simulation"]
    simulation_params = {
        "D_u": sim_params["D_u"],
        "D_v": sim_params["D_v"],
        "k1": sim_params["k1"],
        "k2": sim_params["k2"],
        "domain_size": tuple(sim_params["domain_size"]),
        "resolution": tuple(sim_params["resolution"]),
        "t_end": sim_params["t_end"],
        "dt": sim_params["dt"],
        "save_interval": sim_params["save_interval"],
    }

    print("\nSimulation parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")
    print(
        f"  Steady state: u* = {simulation_params['k1']:.4f}, v* = {simulation_params['k2'] / simulation_params['k1']:.4f}"
    )

    # Extract IC configurations
    ic_configs = config["initial_conditions"]
    print(f"\nInitial conditions: {len(ic_configs)} configurations")

    # Create output directory
    sim_name = config.get("simulation", {}).get("name") or config.get(
        "output_dir", "brusselator"
    )
    base_dir = config.get("output", {}).get("base_dir", "data/datasets")
    data_dir = Path(__file__).parent.parent / base_dir / sim_name
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {data_dir}")

    # Setup coordinate arrays
    domain_size = simulation_params["domain_size"]
    resolution = simulation_params["resolution"]
    x = np.linspace(0, domain_size[0], resolution[1])
    y = np.linspace(0, domain_size[1], resolution[0])

    # Build work items for processing
    work_items = [
        (
            ic_config,
            simulation_params,
            str(data_dir),
            ic_idx,
            x,
            y,
        )
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

            # Check for task-specific parameter overrides
            task_sim_params = simulation_params.copy()

            # Handle parameter sampling from ranges
            # Single RNG per task — draws are sequential so D_u, D_v, k1, k2 get distinct values
            param_rng = np.random.default_rng(ic_config.get("seed"))
            for param in ["D_u", "D_v", "k1", "k2"]:
                raw_val = ic_config.get(param, task_sim_params[param])
                if isinstance(raw_val, list):
                    if len(raw_val) != 2:
                        raise ValueError(
                            f"Task '{ic_name}': {param} as list must have exactly 2 elements [min, max], "
                            f"got {len(raw_val)} elements: {raw_val}"
                        )
                    task_sim_params[param] = param_rng.uniform(raw_val[0], raw_val[1])
                    print(
                        f"Sampled {param} = {task_sim_params[param]:.6f} from range {raw_val}"
                    )
                elif raw_val != task_sim_params[param]:
                    task_sim_params[param] = raw_val

            # k2_delta: sample k2 relative to Turing threshold
            k2_delta_range = ic_config.get("k2_delta")
            k2_c = 0
            if k2_delta_range is not None:
                k2_c = compute_turing_threshold(
                    k1=task_sim_params["k1"],
                    D_u=task_sim_params["D_u"],
                    D_v=task_sim_params["D_v"],
                )
                delta = param_rng.uniform(k2_delta_range[0], k2_delta_range[1])
                task_sim_params["k2"] = k2_c + delta
                print(
                    f"k2_delta mode: k2_c={k2_c:.4f}, delta={delta:.4f}, k2={task_sim_params['k2']:.4f}"
                )

            # Inject k1, k2 into IC config for steady state calculation
            ic_config_with_params = ic_config.copy()
            ic_config_with_params["k1"] = task_sim_params["k1"]
            ic_config_with_params["k2"] = task_sim_params["k2"]

            # Retry loop for divergence
            max_retries = 800
            base_seed = ic_config.get("seed", None)

            for attempt in range(max_retries):
                ic_config_attempt = ic_config_with_params.copy()
                if base_seed is not None:
                    ic_config_attempt["seed"] = base_seed + attempt * 1000
                    if attempt > 0:
                        print(
                            f"  Retry {attempt}/{max_retries - 1} with seed {ic_config_attempt['seed']}"
                        )

                try:
                    # Create initial condition
                    u_init, v_init, _ = create_brusselator_ic( # u_init, v_init, generated_params
                        ic_config_attempt, x, y
                    )

                    # Solve Brusselator
                    concentration_history, times, x_result, y_result = (
                        solve_brusselator(
                            initial_concentration=(u_init, v_init),
                            D_u=task_sim_params["D_u"],
                            D_v=task_sim_params["D_v"],
                            k1=task_sim_params["k1"],
                            k2=task_sim_params["k2"],
                            domain_size=task_sim_params["domain_size"],
                            t_end=task_sim_params["t_end"],
                            dt=task_sim_params["dt"],
                            save_interval=task_sim_params.get("save_interval"),
                        )
                    )

                    # Validate raw concentrations for divergence
                    max_magnitude = 1e6
                    last_u, last_v = concentration_history[-1]
                    for label, arr in [("u", last_u), ("v", last_v)]:
                        max_val = np.abs(arr).max()
                        if (
                            max_val > max_magnitude
                            or np.any(np.isnan(arr))
                            or np.any(np.isinf(arr))
                        ):
                            raise ValueError(
                                f"Silent divergence: {label} has max magnitude {max_val:.2e}"
                            )

                    ic_config_to_save = ic_config.copy()
                    ic_config_to_save["k1_used"] = task_sim_params["k1"]
                    ic_config_to_save["k2_used"] = task_sim_params["k2"]
                    ic_config_to_save["D_u_used"] = task_sim_params["D_u"]
                    ic_config_to_save["D_v_used"] = task_sim_params["D_v"]
                    ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
                    ic_config_to_save["retry_attempt"] = attempt
                    if k2_delta_range is not None:
                        assert k2_c > 0
                        ic_config_to_save["k2_c"] = k2_c
                        ic_config_to_save["k2_delta_used"] = (
                            task_sim_params["k2"] - k2_c
                        )

                    fourier_data = generate_fourier_data(
                        concentration_history, times
                    )
                    np.savez(
                        output_file,
                        **fourier_data,
                        ic_config=ic_config_to_save,
                        simulation_params=task_sim_params,  # type: ignore[reportArgumentType]
                    )
                    print(f"\nSaved Fourier data to {output_file}")

                    if attempt > 0:
                        print(f"  (succeeded after {attempt} retries)")

                    # Generate visualization
                    vis_file = data_dir / f"{ic_name}_evolution.png"
                    save_brusselator_evolution(
                        concentration_history,
                        times,
                        x_result,
                        y_result,
                        str(vis_file),
                        n_snapshots=4,
                    )

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
                    if (
                        any(
                            kw in error_str
                            for kw in ["diverge", "nan", "inf", "overflow"]
                        )
                        and attempt < max_retries - 1
                    ):
                        print(f"  ⚠️  {type(e).__name__}: {str(e)}, will retry...")
                        continue
                    print(f"\n⚠️  SKIPPED: {ic_name}")
                    print(f"    Error: {type(e).__name__}: {str(e)}")
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
        print("  - Fourier coefficients: u_hat, v_hat (complex128)")
        print("  - Metadata: ic_config, simulation_params")


if __name__ == "__main__":
    main()
