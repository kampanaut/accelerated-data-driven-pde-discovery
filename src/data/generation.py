"""
Shared data generation framework for PDE discovery.

Provides PDESpec (a dataclass declaring PDE-specific behavior) and
run_generation() (the orchestration engine). Each PDE script becomes
a thin wrapper: define a PDESpec, call run_generation().

Pickling note: process_single_ic is a module-level function and all
callables stored in PDESpec must also be module-level so that the
multiprocessing 'spawn' context can serialize work items.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# FieldType enum
# ---------------------------------------------------------------------------


class FieldType(Enum):
    SCALAR = "scalar"
    PAIRED = "paired"


# ---------------------------------------------------------------------------
# PDESpec dataclass
# ---------------------------------------------------------------------------


@dataclass
class PDESpec:
    """Declares everything PDE-specific as data and callables."""

    name: str
    pde_param_keys: list[str]
    create_ic: Callable  # (ic_config, x, y) -> (u, params) or (u, v, params)
    solve: Callable  # (initial_fields, sim_params, task_name) -> dict
    field_type: FieldType = FieldType.SCALAR
    default_output_name: str = "pde"

    # Which PDE params support [min, max] sampling in ic_config.
    # None means all pde_param_keys are samplable.
    samplable_params: list[str] | None = None

    # Key in solver result dict that holds the field history.
    history_key: str = "field_history"

    # Hook: mutate ic_config before IC creation (e.g. BR injects k1/k2, LO injects a).
    prepare_ic_config: Callable | None = None  # (ic_config, sim_params) -> ic_config

    # Hook: mutate sim_params after sampling (e.g. BR k2_delta mode).
    post_sample_params: Callable | None = (
        None  # (ic_config, sim_params, rng) -> sim_params
    )

    # Hook: post-process fourier_data after FFT (e.g. NS adds p_hat from solver results).
    # Signature: (fourier_data, solver_results) -> fourier_data
    post_fourier: Callable | None = None

    # Extra keys to save in .npz beyond fourier_data + ic_config + simulation_params.
    # Each entry is (npz_key, param_key) — the value is pulled from task_sim_params.
    extra_save_keys: list[tuple[str, str]] = field(default_factory=list)

    # Physical magnitude check on last snapshot (BR uses 1e6).
    max_physical_magnitude: float | None = None

    # BR catches broader exception patterns (nan/inf/overflow/diverge in any Exception).
    broad_divergence_catch: bool = False

    # Visualization callback.  Signature for most PDEs:
    #   (field_history, times, x, y, path, n_snapshots=8) -> None
    # NS additionally needs dx, dy — see vis_needs_grid_spacing.
    save_visualization: Callable | None = None
    vis_needs_grid_spacing: bool = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Simulation parameter extraction
# ---------------------------------------------------------------------------


def extract_simulation_params(config: dict, spec: PDESpec) -> dict:
    """Pull universal + PDE-specific keys from config['simulation']."""
    sim = config["simulation"]
    params: dict[str, Any] = {
        "domain_size": tuple(sim["domain_size"]),
        "resolution": tuple(sim["resolution"]),
        "t_end": sim["t_end"],
        "dt": sim["dt"],
        "save_interval": sim["save_interval"],
    }
    for key in spec.pde_param_keys:
        params[key] = sim[key]
    return params


# ---------------------------------------------------------------------------
# Per-task parameter sampling
# ---------------------------------------------------------------------------


def sample_params(
    ic_config: dict,
    sim_params: dict,
    spec: PDESpec,
    rng: np.random.Generator,
) -> dict:
    """
    Sample PDE parameters that have [min, max] ranges in ic_config.

    Returns a shallow copy of sim_params with sampled values filled in.
    """
    task_sim_params = sim_params.copy()
    samplable = (
        spec.samplable_params
        if spec.samplable_params is not None
        else spec.pde_param_keys
    )

    for key in samplable:
        raw = ic_config.get(key, task_sim_params[key])
        if isinstance(raw, list):
            task_sim_params[key] = rng.uniform(raw[0], raw[1])
        elif raw != task_sim_params[key]:
            task_sim_params[key] = raw

    # For non-samplable PDE params, still allow ic_config overrides (fixed values).
    if spec.samplable_params is not None:
        for key in spec.pde_param_keys:
            if key not in samplable:
                raw = ic_config.get(key, task_sim_params[key])
                if not isinstance(raw, list) and raw != task_sim_params[key]:
                    task_sim_params[key] = raw

    if spec.post_sample_params is not None:
        task_sim_params = spec.post_sample_params(ic_config, task_sim_params, rng)

    return task_sim_params


# ---------------------------------------------------------------------------
# Fourier transform
# ---------------------------------------------------------------------------


def generate_fourier_data(
    field_history: list,
    times: np.ndarray,
    field_type: FieldType,
) -> dict:
    """FFT field snapshots, dispatching on FieldType."""

    if field_type == FieldType.SCALAR:
        n = len(field_history)
        ny, nx = field_history[0].shape
        u_hat = np.empty((n, ny, nx), dtype=np.complex128)
        for i, u in enumerate(field_history):
            u_hat[i] = np.fft.fft2(u)
        print(f"  FFT'd {n} snapshots, shape ({ny}, {nx})")
        return {"u_hat": u_hat, "times": times}

    elif field_type == FieldType.PAIRED:
        n = len(field_history)
        ny, nx = field_history[0][0].shape
        u_hat = np.empty((n, ny, nx), dtype=np.complex128)
        v_hat = np.empty((n, ny, nx), dtype=np.complex128)
        for i, (u, v) in enumerate(field_history):
            u_hat[i] = np.fft.fft2(u)
            v_hat[i] = np.fft.fft2(v)
        print(f"  FFT'd {n} snapshots, shape ({ny}, {nx})")
        return {"u_hat": u_hat, "v_hat": v_hat, "times": times}

    raise ValueError(f"Unknown field type: {field_type}")


# ---------------------------------------------------------------------------
# Fourier validation
# ---------------------------------------------------------------------------


def validate_fourier_data(fourier_data: dict, field_type: FieldType) -> None:
    """Raise ValueError if any Fourier arrays contain NaN/Inf."""
    if field_type == FieldType.SCALAR:
        keys = ["u_hat"]
    elif field_type == FieldType.PAIRED:
        keys = ["u_hat", "v_hat"]
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    for key in keys:
        arr = fourier_data[key]
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"Silent divergence: {key} has NaN/Inf")


# ---------------------------------------------------------------------------
# Physical magnitude validation
# ---------------------------------------------------------------------------


def validate_physical_magnitude(
    field_history: list,
    field_type: FieldType,
    max_magnitude: float,
) -> None:
    """Check last snapshot for blown-up values. Raises ValueError."""
    last = field_history[-1]

    if field_type == FieldType.SCALAR:
        arrays = [("u", last)]
    else:
        arrays = [("u", last[0]), ("v", last[1])]

    for label, arr in arrays:
        max_val = np.abs(arr).max()
        if max_val > max_magnitude or np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(
                f"Silent divergence: {label} has max magnitude {max_val:.2e}"
            )


# ---------------------------------------------------------------------------
# Worker function (must be module-level for pickling)
# ---------------------------------------------------------------------------


def process_single_ic(args_tuple):
    """
    Worker for multiprocessing Pool.

    Args:
        args_tuple: (spec, ic_config, simulation_params, data_dir, ic_idx, x, y)

    Returns:
        (status, ic_name, error_msg, retry_count)
    """
    spec, ic_config, simulation_params, data_dir, ic_idx, x, y = args_tuple

    ic_name = ic_config.get("name", f"ic_{ic_idx}")

    output_file = Path(data_dir) / f"{ic_name}_fourier.npz"
    if output_file.exists():
        return ("skipped", ic_name, None, 0)

    max_retries = 800
    base_seed = ic_config.get("seed", None)

    for attempt in range(max_retries):
        rng = np.random.default_rng(
            base_seed + attempt * 1000 if base_seed is not None else None
        )

        # Sample parameters
        task_sim_params = sample_params(ic_config, simulation_params, spec, rng)

        # Prepare IC config (hooks like BR k1/k2 injection, LO a injection)
        ic_config_attempt = ic_config.copy()
        if spec.prepare_ic_config is not None:
            ic_config_attempt = spec.prepare_ic_config(
                ic_config_attempt, task_sim_params
            )

        if base_seed is not None:
            ic_config_attempt["seed"] = base_seed + attempt * 1000

        try:
            # Create initial condition
            ic_result = spec.create_ic(ic_config_attempt, x, y)

            # Build initial_fields tuple for solver
            if spec.field_type == FieldType.SCALAR:
                u_init, _ = ic_result
                initial_fields = (u_init,)
            else:
                u_init, v_init, _ = ic_result
                initial_fields = (u_init, v_init)

            # Solve
            results = spec.solve(initial_fields, task_sim_params, task_name=ic_name)

            field_history = results[spec.history_key]
            times = results["times"]

            # Physical magnitude check
            if spec.max_physical_magnitude is not None:
                validate_physical_magnitude(
                    field_history, spec.field_type, spec.max_physical_magnitude
                )

            # FFT
            fourier_data = generate_fourier_data(field_history, times, spec.field_type)
            validate_fourier_data(fourier_data, spec.field_type)

            # Post-FFT hook (e.g. NS adds p_hat)
            if spec.post_fourier is not None:
                fourier_data = spec.post_fourier(fourier_data, results)

            # Build save metadata
            ic_config_to_save = ic_config.copy()
            ic_config_to_save["seed_used"] = ic_config_attempt.get("seed")
            ic_config_to_save["retry_attempt"] = attempt

            # Record sampled parameter values
            samplable = (
                spec.samplable_params
                if spec.samplable_params is not None
                else spec.pde_param_keys
            )
            for key in samplable:
                ic_config_to_save[f"{key}_used"] = task_sim_params[key]

            # Extra save keys
            extra_kwargs = {}
            for npz_key, param_key in spec.extra_save_keys:
                extra_kwargs[npz_key] = task_sim_params[param_key]

            np.savez(
                output_file,
                **fourier_data,
                ic_config=ic_config_to_save,
                simulation_params=task_sim_params,  # type: ignore[reportArgumentType]
                **extra_kwargs,
            )

            # Visualization
            if spec.save_visualization is not None:
                vis_file = Path(data_dir) / f"{ic_name}_evolution.png"
                if spec.vis_needs_grid_spacing:
                    x_res = results.get("x", x)
                    y_res = results.get("y", y)
                    dx = float(x_res[1] - x_res[0])
                    dy = float(y_res[1] - y_res[0])
                    spec.save_visualization(
                        field_history, times, x_res, y_res, dx, dy, str(vis_file)
                    )
                else:
                    spec.save_visualization(field_history, times, x, y, str(vis_file))

            return ("success", ic_name, None, attempt)

        except RuntimeError:
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
            if spec.broad_divergence_catch:
                error_str = str(e).lower()
                if (
                    any(kw in error_str for kw in ["diverge", "nan", "inf", "overflow"])
                    and attempt < max_retries - 1
                ):
                    continue
            return ("failed", ic_name, f"{type(e).__name__}: {str(e)}", attempt)

    return ("failed", ic_name, f"Exhausted {max_retries} retries", max_retries)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_generation(spec: PDESpec, config_path: str | Path, workers: int = 1) -> None:
    """
    Full data generation pipeline.

    Args:
        spec: PDESpec declaring PDE-specific behavior.
        config_path: Path to YAML configuration.
        workers: Number of parallel workers.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    print("=" * 60)
    print(f"{spec.name} Data Generation for PDE Discovery")
    print("=" * 60)
    print(f"Config file: {config_path}")

    simulation_params = extract_simulation_params(config, spec)

    print("\nSimulation parameters:")
    for key, value in simulation_params.items():
        print(f"  {key}: {value}")

    ic_configs = config["initial_conditions"]
    print(f"\nInitial conditions: {len(ic_configs)} configurations")

    # Uniform coefficient sampling: pre-sample N values and assign one per task.
    # Per-task fixed scalars are preserved. Per-task [min, max] ranges are used
    # for that task's sampling range. Global range is the fallback.
    if config.get("simulation", {}).get("uniform_sample", False):
        samplable = (
            spec.samplable_params
            if spec.samplable_params is not None
            else spec.pde_param_keys
        )
        sample_seed = config.get("simulation", {}).get("seed", 42)
        rng = np.random.default_rng(sample_seed)
        for key in samplable:
            # Collect tasks that have a [min, max] range for this key
            range_tasks = []
            for i, ic in enumerate(ic_configs):
                raw = ic.get(key, simulation_params.get(key))
                if isinstance(raw, list) and len(raw) == 2:
                    range_tasks.append((i, raw[0], raw[1]))
            if range_tasks:
                # Sample one value per task from its own range
                values = [rng.uniform(lo, hi) for _, lo, hi in range_tasks]
                values.sort()
                print(f"\n  Uniform sample for '{key}': {len(range_tasks)} tasks")
                for (idx, lo, hi), val in zip(range_tasks, values):
                    ic_configs[idx][key] = float(val)
                    print(f"    {ic_configs[idx].get('name', idx)}: {key}={val:.6f} (from [{lo}, {hi}])")

    # Output directory
    sim_name = config.get("simulation", {}).get("name") or config.get(
        "output_dir", spec.default_output_name
    )
    base_dir = config.get("output", {}).get("base_dir", "data/datasets")
    data_dir = Path(__file__).parent.parent.parent / base_dir / sim_name
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {data_dir}")

    # Coordinate arrays
    domain_size = simulation_params["domain_size"]
    resolution = simulation_params["resolution"]
    x = np.linspace(0, domain_size[0], resolution[1], endpoint=False)
    y = np.linspace(0, domain_size[1], resolution[0], endpoint=False)

    # Build work items
    work_items = [
        (spec, ic_config, simulation_params, str(data_dir), ic_idx, x, y)
        for ic_idx, ic_config in enumerate(ic_configs)
    ]

    if workers > 1:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        print(f"\nUsing {workers} parallel workers")

        results = []
        with ctx.Pool(workers) as pool:
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
        _print_dataset_contents(spec.field_type)


def _print_dataset_contents(field_type: FieldType) -> None:
    """Print what each dataset file contains."""
    print("\nEach dataset contains:")
    if field_type == FieldType.SCALAR:
        print("  - Fourier coefficients: u_hat (complex128)")
    elif field_type == FieldType.PAIRED:
        print("  - Fourier coefficients: u_hat, v_hat (complex128)")
    print("  - Metadata: ic_config, simulation_params")
