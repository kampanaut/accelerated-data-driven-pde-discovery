"""
Generate heat equation training data for PDE discovery.

This script:
1. Loads IC configuration from YAML file
2. Solves heat equation u_t = D*nabla^2(u) for each IC
3. FFTs all snapshots to Fourier coefficients
4. Saves .npz files (Fourier-only)

Targets come from PDE RHS (not temporal FD), so we store
ALL snapshots and don't need u_t_hat.

Usage:
    python scripts/generate_heat_data.py --config configs/heat_train-1.yaml
    python scripts/generate_heat_data.py --config configs/heat_train-1.yaml --gpu
"""

import sys
import os
from pathlib import Path

# Default to CPU for stability
if "--gpu" in sys.argv:
    sys.argv.remove("--gpu")
    os.environ.pop("JAX_PLATFORMS", None)
else:
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import argparse
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pde.heat_equation import solve_heat_with_params

try:
    from phiml.math._optimize import Diverged
except ImportError:
    Diverged = Exception


def generate_fourier_data(
    field_history: list, times: np.ndarray
) -> dict:
    """
    Convert field snapshots to Fourier coefficients.

    Args:
        field_history: List of 2D numpy arrays (scalar field at each time)
        times: Time values for each snapshot

    Returns:
        Dict with keys: u_hat (complex128), times (float64)
    """
    n_snapshots = len(field_history)
    ny, nx = field_history[0].shape

    u_hat = np.empty((n_snapshots, ny, nx), dtype=np.complex128)

    for i, u in enumerate(field_history):
        u_hat[i] = np.fft.fft2(u)

    print(f"  FFT'd {n_snapshots} snapshots, shape ({ny}, {nx})")

    return {
        "u_hat": u_hat,
        "times": times,
    }


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_single_ic(args_tuple):
    """
    Worker function for multiprocessing Pool.

    Args:
        args_tuple: (ic_config, simulation_params, data_dir, ic_idx, x, y)

    Returns:
        (status, ic_name, error_msg, retry_count)
    """
    ic_config, simulation_params, data_dir, ic_idx, x, y = args_tuple

    ic_name = ic_config.get("name", f"ic_{ic_idx}")

    output_file = Path(data_dir) / f"{ic_name}_fourier.npz"
    if output_file.exists():
        return ("skipped", ic_name, None, 0)

    rng = np.random.default_rng(ic_config.get("seed"))

    task_sim_params = simulation_params.copy()

    # D: scalar or [min, max] range
    raw_D = ic_config.get("D", simulation_params["D"])
    if isinstance(raw_D, list):
        task_D = rng.uniform(raw_D[0], raw_D[1])
    else:
        task_D = raw_D
    task_sim_params["D"] = task_D

    max_retries = 5
    base_seed = ic_config.get("seed", None)

    for attempt in range(max_retries):
        ic_config_attempt = ic_config.copy()
        if base_seed is not None:
            ic_config_attempt["seed"] = base_seed + attempt * 1000

        try:
            from src.data.initial_conditions_heat import create_heat_ic

            u_init, _ = create_heat_ic(ic_config_attempt, x, y)

            ic_params_for_solver = {
                "type": "custom",
                "u_init": u_init,
            }

            results = solve_heat_with_params(ic_params_for_solver, task_sim_params)

            field_history = results["field_history"]
            times = results["times"]

            ic_config_to_save = ic_config.copy()
            ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
            ic_config_to_save["retry_attempt"] = attempt
            ic_config_to_save["D_used"] = task_D

            fourier_data = generate_fourier_data(field_history, times)

            # Validate
            arr = fourier_data["u_hat"]
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise ValueError("Silent divergence: u_hat has NaN/Inf")

            np.savez(
                output_file,
                **fourier_data,
                ic_config=ic_config_to_save,
                simulation_params=task_sim_params,
                D_used=task_D,
            )

            return ("success", ic_name, None, attempt)

        except Diverged:
            if attempt < max_retries - 1:
                continue
            return ("failed", ic_name, f"Diverged after {max_retries} attempts", max_retries)

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
    parser = argparse.ArgumentParser(description="Generate heat equation training data")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("=" * 60)
    print("Heat Equation Data Generation for PDE Discovery")
    print("=" * 60)
    print(f"Config file: {config_path}")

    sim_params = config["simulation"]
    simulation_params = {
        "D": sim_params["D"],
        "domain_size": tuple(sim_params["domain_size"]),
        "resolution": tuple(sim_params["resolution"]),
        "t_end": sim_params["t_end"],
        "dt": sim_params["dt"],
        "save_interval": sim_params["save_interval"],
    }

    print("\nSimulation parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")

    ic_configs = config["initial_conditions"]
    print(f"\nInitial conditions: {len(ic_configs)} configurations")

    sim_name = config.get("simulation", {}).get("name") or config.get("output_dir", "heat")
    base_dir = config.get("output", {}).get("base_dir", "data/datasets")
    data_dir = Path(__file__).parent.parent / base_dir / sim_name
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {data_dir}")

    domain_size = simulation_params["domain_size"]
    resolution = simulation_params["resolution"]
    x = np.linspace(0, domain_size[0], resolution[1])
    y = np.linspace(0, domain_size[1], resolution[0])

    work_items = [
        (ic_config, simulation_params, str(data_dir), ic_idx, x, y)
        for ic_idx, ic_config in enumerate(ic_configs)
    ]

    if args.workers > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")

        print(f"\nUsing {args.workers} parallel workers")

        results = []
        with ctx.Pool(args.workers) as pool:
            for result in pool.imap_unordered(process_single_ic, work_items):
                status, name, error, retries = result
                results.append(result)
                done = len(results)
                if status == "success":
                    retry_info = f" (after {retries} retries)" if retries > 0 else ""
                    print(f"[{done}/{len(ic_configs)}] {name}: SUCCESS{retry_info}")
                elif status == "skipped":
                    print(f"[{done}/{len(ic_configs)}] {name}: SKIPPED (exists)")
                else:
                    print(f"[{done}/{len(ic_configs)}] {name}: FAILED - {error}")

        successful = sum(1 for r in results if r[0] in ("success", "skipped"))
        skipped_existing = sum(1 for r in results if r[0] == "skipped")
        failed_tasks = [(r[1], r[2]) for r in results if r[0] == "failed"]

    else:
        successful = 0
        failed_tasks = []
        skipped_existing = 0

        for ic_idx, ic_config in enumerate(ic_configs):
            result = process_single_ic(work_items[ic_idx])
            status, name, error, retries = result

            if status == "success":
                retry_info = f" (after {retries} retries)" if retries > 0 else ""
                print(f"[{ic_idx + 1}/{len(ic_configs)}] {name}: SUCCESS{retry_info}")
                successful += 1
            elif status == "skipped":
                print(f"[{ic_idx + 1}/{len(ic_configs)}] {name}: SKIPPED (exists)")
                skipped_existing += 1
                successful += 1
            else:
                print(f"[{ic_idx + 1}/{len(ic_configs)}] {name}: FAILED - {error}")
                failed_tasks.append((name, error))

    print(f"\n{'=' * 60}")
    print("Data generation complete!")
    print(f"{'=' * 60}")
    print(f"\nSummary:")
    print(f"  Total tasks: {len(ic_configs)}")
    print(f"  Successful: {successful} ({skipped_existing} already existed)")
    print(f"  Failed: {len(failed_tasks)}")
    print(f"\nOutput directory: {data_dir}")

    if failed_tasks:
        print("\nFailed tasks:")
        for name, error in failed_tasks:
            print(f"    - {name}: {error}")

    if successful > 0:
        print("\nEach dataset contains:")
        print("  - Fourier coefficients: u_hat (complex128)")
        print("  - Metadata: ic_config, simulation_params")


if __name__ == "__main__":
    main()
