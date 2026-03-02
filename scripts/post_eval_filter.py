"""Post-evaluation model filtering.

Scans results.json across all formal experiments, flags models with
full-range coefficient failure (any task, any combo), and reports
partial failures for kept models.

Usage:
    uv run python scripts/post_eval_filter.py [--pde heat|br|nl_heat]
"""

import argparse
import json
from pathlib import Path

import yaml

from src.evaluation.metrics import compress_step_ranges


def load_experiment(exp_dir: Path) -> tuple[dict, list[int]] | None:
    """Load results.json and fixed_steps from an experiment directory."""
    results_path = exp_dir / "evaluation" / "results.json"
    config_path = exp_dir / "training" / "config.yaml"

    if not results_path.exists() or not config_path.exists():
        return None

    with open(results_path) as f:
        results = json.load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    fixed_steps = config.get("evaluation", {}).get("fixed_steps", [])
    return results, fixed_steps


def check_full_failure(
    coeff_steps: dict[str, list[int]], fixed_steps: list[int]
) -> dict[str, bool]:
    """Check which coefficients have full-range failure."""
    fixed_set = set(fixed_steps)
    return {
        name: set(steps) >= fixed_set
        for name, steps in coeff_steps.items()
        if steps
    }


def analyze_experiment(
    exp_name: str, results: dict, fixed_steps: list[int]
) -> tuple[bool, dict]:
    """Analyze one experiment. Returns (is_flagged, report_data)."""
    tasks = results.get("tasks", {})
    flagged = False
    n_tasks = len(tasks)
    n_clean = 0

    task_reports: dict[str, dict] = {}
    # Track which (task, combo, coeff, range_str) triggered full-range failure
    full_failure_entries: list[tuple[str, str, str, str]] = []

    for task_name, task_data in sorted(tasks.items()):
        task_coeff_flags: dict[str, list[tuple[str, str]]] = {}  # combo -> [(coeff, range_str)]
        task_loss_flags: dict[str, str] = {}  # combo -> range_str
        task_has_any_flag = False

        # Find all combo keys
        combo_keys = [
            k.removeprefix("worse_")
            for k in task_data
            if k.startswith("worse_")
        ]

        for combo_key in sorted(combo_keys):
            worse = task_data[f"worse_{combo_key}"]
            coeff_steps = worse.get("coeff_steps", {})
            loss_steps = worse.get("loss_steps", [])

            # Check for full failure
            full_failures = check_full_failure(coeff_steps, fixed_steps)
            for coeff_name, is_full in full_failures.items():
                if is_full:
                    flagged = True
                    range_str = compress_step_ranges(coeff_steps[coeff_name], fixed_steps)
                    full_failure_entries.append((task_name, combo_key, coeff_name, range_str))

            # Collect coeff flags for this combo
            for coeff_name, steps in coeff_steps.items():
                if steps:
                    task_has_any_flag = True
                    range_str = compress_step_ranges(steps, fixed_steps)
                    if combo_key not in task_coeff_flags:
                        task_coeff_flags[combo_key] = []
                    task_coeff_flags[combo_key].append((coeff_name, range_str))

            # Collect loss flags for this combo
            if loss_steps:
                task_has_any_flag = True
                task_loss_flags[combo_key] = compress_step_ranges(
                    loss_steps, fixed_steps
                )

        if task_has_any_flag:
            task_reports[task_name] = {
                "coeff": task_coeff_flags,
                "loss": task_loss_flags,
            }
        else:
            n_clean += 1

    return flagged, {
        "n_tasks": n_tasks,
        "n_clean": n_clean,
        "task_reports": task_reports,
        "full_failure_entries": full_failure_entries,
    }


def print_report(
    kept: list[tuple[str, dict]],
    filtered: list[tuple[str, dict]],
) -> None:
    """Print the filtering report."""
    # Filtered models — only show the full-failure combos that triggered filtering
    if filtered:
        print("=" * 60)
        print(f"FILTERED OUT ({len(filtered)} models — full-range coeff failure)")
        print("=" * 60)
        for exp_name, data in filtered:
            entries = data["full_failure_entries"]
            n_tasks = data["n_tasks"]
            # Group by (task, coeff) → collect combo keys
            task_coeff: dict[tuple[str, str], list[str]] = {}
            for task_name, combo_key, coeff_name, _ in entries:
                key = (task_name, coeff_name)
                if key not in task_coeff:
                    task_coeff[key] = []
                task_coeff[key].append(combo_key)

            n_failed_tasks = len({t for t, _ in task_coeff})
            print(f"\n  {exp_name}  ({n_failed_tasks}/{n_tasks} tasks failed)")
            for (task_name, coeff_name), combo_keys in sorted(task_coeff.items()):
                # Extract unique K and noise values from combo keys
                k_vals: set[str] = set()
                noise_vals: set[str] = set()
                for ck in combo_keys:
                    parts = ck.split("_")
                    # combo_key format: k_{val}_noise_{val}
                    k_vals.add(parts[1])
                    noise_vals.add(parts[3])
                k_sorted = sorted(k_vals, key=lambda x: int(x))
                noise_sorted = sorted(noise_vals, key=lambda x: float(x))
                print(
                    f"    {task_name}  {coeff_name}  "
                    f"K=[{','.join(k_sorted)}] "
                    f"noise=[{','.join(noise_sorted)}]"
                )
        print()

    # Kept models
    print("=" * 60)
    print(f"KEPT ({len(kept)} models)")
    print("=" * 60)
    for exp_name, data in kept:
        n_clean = data["n_clean"]
        n_tasks = data["n_tasks"]
        print(f"\n{exp_name}  {n_clean}/{n_tasks} clean")

        # Coeff flags
        coeff_tasks = {
            t: r["coeff"]
            for t, r in data["task_reports"].items()
            if r["coeff"]
        }
        if coeff_tasks:
            print("  coeff:")
            for task_name, combos in sorted(coeff_tasks.items()):
                print(f"    {task_name}")
                for combo_key, coeff_flags in sorted(combos.items()):
                    flags_str = "  ".join(
                        f"{n}@[{r}]" for n, r in coeff_flags
                    )
                    print(f"      {combo_key}    {flags_str}")

        # Loss flags
        loss_tasks = {
            t: r["loss"]
            for t, r in data["task_reports"].items()
            if r["loss"]
        }
        if loss_tasks:
            print("  loss:")
            for task_name, combos in sorted(loss_tasks.items()):
                print(f"    {task_name}")
                for combo_key, range_str in sorted(combos.items()):
                    print(f"      {combo_key}    loss@[{range_str}]")


def main():
    parser = argparse.ArgumentParser(description="Post-evaluation model filtering")
    parser.add_argument(
        "--pde",
        choices=["heat", "br", "nl_heat"],
        help="Filter by PDE type (default: all)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/models/formal"),
        help="Base directory to scan (default: data/models/formal)",
    )
    args = parser.parse_args()

    formal_dir: Path = args.dir
    if not formal_dir.exists():
        print(f"No formal experiments found at {formal_dir}")
        return

    kept: list[tuple[str, dict]] = []
    filtered: list[tuple[str, dict]] = []
    skipped = 0

    # Collect experiment dirs — handle both flat and nested layouts
    exp_dirs: list[Path] = []
    for child in sorted(formal_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "training" / "config.yaml").exists():
            # Flat: experiments directly in formal_dir
            if not args.pde:
                exp_dirs.append(child)
        else:
            # Nested: pde_dir/exp_dir
            if args.pde and child.name != args.pde:
                continue
            for exp_dir in sorted(child.iterdir()):
                if exp_dir.is_dir():
                    exp_dirs.append(exp_dir)

    for exp_dir in exp_dirs:
        result = load_experiment(exp_dir)
        if result is None:
            skipped += 1
            continue

        results, fixed_steps = result
        if not fixed_steps:
            skipped += 1
            continue

        exp_name = exp_dir.name
        is_flagged, report_data = analyze_experiment(
            exp_name, results, fixed_steps
        )

        if is_flagged:
            filtered.append((exp_name, report_data))
        else:
            kept.append((exp_name, report_data))

    if skipped:
        print(f"({skipped} experiments skipped — no results.json or config.yaml)\n")

    print_report(kept, filtered)


if __name__ == "__main__":
    main()
