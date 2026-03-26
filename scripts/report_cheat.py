"""Generate cheat experiment report logs.

Reads results.json and weight NPZs from cheat experiment variants.
Produces per-variant logs and a cross-variant summary.

Usage:
    uv run python scripts/report_cheat.py --model-type cheat \
        --variants data/models/cheat/heat data/models/cheat_finn/heat ...
    uv run python scripts/report_cheat.py --model-type cheat --auto
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


def load_model_results(model_dir: Path) -> Optional[dict]:
    """Load results.json and config for one model variant."""
    results_path = model_dir / "evaluation" / "results.json"
    config_path = model_dir / "training" / "config.yaml"

    if not results_path.exists():
        return None

    with open(results_path) as f:
        results = json.load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {"results": results, "config": config, "dir": model_dir}


def load_theta_star_weights(model_dir: Path) -> Optional[np.ndarray]:
    """Load θ* weights from best_model.pt checkpoint."""
    best_path = model_dir / "checkpoints" / "best_model.pt"
    if not best_path.exists():
        return None
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    weight_keys = [k for k in state if "weight" in k]
    if len(weight_keys) == 1:
        return state[weight_keys[0]].numpy().flatten()
    return None


def load_theta0_weights(model_dir: Path) -> Optional[np.ndarray]:
    """Load θ₀ weights from initial_model.pt checkpoint."""
    init_path = model_dir / "checkpoints" / "initial_model.pt"
    if not init_path.exists():
        return None
    ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    weight_keys = [k for k in state if "weight" in k]
    if len(weight_keys) == 1:
        return state[weight_keys[0]].numpy().flatten()
    return None


def load_weight_trajectory(model_dir: Path, task_name: str, combo_key: str,
                           method: str) -> Optional[np.ndarray]:
    """Load weight trajectory from evaluation NPZ. Returns (n_steps, n_params)."""
    npz_path = model_dir / "evaluation" / "samples" / f"{task_name}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    key = f"{combo_key}/{method}/weights"
    if key in data:
        return data[key]
    return None


def get_recovery_at_step(task_data: dict, combo_key: str, method: str,
                         step_idx: int) -> Optional[dict]:
    """Get recovered D and error at a specific step index for one combo."""
    cr_key = f"coefficient_recovery_{combo_key}"
    if cr_key not in task_data:
        return None
    cr = task_data[cr_key]
    m = cr.get(method, {})
    if not m:
        return None

    result = {}
    for key in m:
        if key.endswith("_recovered"):
            coeff_name = key.replace("_recovered", "")
            recovered_list = m[key]
            error_list = m.get(f"{coeff_name}_error_pct", [])
            true_val = m.get(f"{coeff_name}_true")
            if step_idx < len(recovered_list):
                result[coeff_name] = {
                    "true": true_val,
                    "recovered": recovered_list[step_idx],
                    "error_pct": error_list[step_idx] if step_idx < len(error_list) else None,
                }
    return result


def get_noise0_combos(task_data: dict) -> list[str]:
    """Get combo keys with noise=0.00."""
    combos = []
    for key in task_data:
        if key.startswith("coefficient_recovery_") and "noise_0.00" in key:
            combo_key = key.replace("coefficient_recovery_", "")
            combos.append(combo_key)
    return sorted(combos)


def get_designed_step_idx(config: dict) -> int:
    """Get the step index corresponding to inner_steps from config."""
    inner_steps = config["training"]["inner_steps"]
    fixed_steps = config["evaluation"]["fixed_steps"]
    if inner_steps in fixed_steps:
        return fixed_steps.index(inner_steps)
    # Find closest
    for i, s in enumerate(fixed_steps):
        if s >= inner_steps:
            return i
    return len(fixed_steps) - 1


def get_last_step_idx(config: dict) -> int:
    """Get the last step index."""
    return len(config["evaluation"]["fixed_steps"]) - 1


def format_weights(w: np.ndarray, labels: list[str]) -> str:
    """Format weight array with labels."""
    parts = [f"{l}={v:+.4f}" for l, v in zip(labels, w)]
    return "  ".join(parts)


WEIGHT_LABELS_5 = ["w_u", "w_ux", "w_uy", "w_uxx", "w_uyy"]


def write_per_variant_log(variant_dir: Path, model_type: str, out_path: Path):
    """Write per-variant log file."""
    variant_name = variant_dir.parent.name  # e.g., cheat_finn

    # Find all model subdirs
    model_dirs = sorted([
        d for d in variant_dir.iterdir()
        if d.is_dir() and (d / "evaluation" / "results.json").exists()
    ])

    if not model_dirs:
        print(f"  No evaluated models in {variant_dir}")
        return None

    lines = []
    lines.append("=" * 60)
    lines.append(f"CHEAT EXPERIMENT REPORT: {variant_name}")
    lines.append("=" * 60)
    lines.append(f"Model type: {model_type}")
    lines.append(f"Model variants found: {len(model_dirs)}")
    lines.append("")

    # θ* and θ₀ from first model (same across all for cheat)
    if model_type == "cheat":
        theta_star = load_theta_star_weights(model_dirs[0])
        theta0 = load_theta0_weights(model_dirs[0])
        if theta_star is not None:
            lines.append(f"θ* weights: {format_weights(theta_star, WEIGHT_LABELS_5)}")
        if theta0 is not None:
            lines.append(f"θ₀ weights: {format_weights(theta0, WEIGHT_LABELS_5)}")
        lines.append("")

    # Collect per-model summaries for variant aggregate
    all_designed_means = []
    all_last_means = []
    all_split_theta_star = []
    all_split_designed = []
    all_split_last = []
    all_nonrhs_theta_star = []
    all_nonrhs_last = []

    for model_dir in model_dirs:
        data = load_model_results(model_dir)
        if data is None:
            continue

        config = data["config"]
        results = data["results"]
        model_name = model_dir.name
        inner_steps = config["training"]["inner_steps"]
        k_shot = config["training"]["k_shot"]

        # Determine loss mode from name
        loss_mode = "baseline"
        for mode in ["metal-spectral", "metal", "spectral"]:
            if mode in model_name:
                loss_mode = mode
                break

        designed_idx = get_designed_step_idx(config)
        last_idx = get_last_step_idx(config)
        fixed_steps = config["evaluation"]["fixed_steps"]
        designed_step = fixed_steps[designed_idx]
        last_step = fixed_steps[last_idx]

        lines.append(f"--- {model_name} ---")
        lines.append(f"  inner_steps={inner_steps}, k_shot={k_shot}, loss_mode={loss_mode}")
        lines.append("")

        # Per-task recovery at designed and last step
        for step_label, step_idx, step_num, collector in [
            ("designed", designed_idx, designed_step, all_designed_means),
            ("last", last_idx, last_step, all_last_means),
        ]:
            lines.append(f"  Recovery at step {step_num} ({step_label}):  [noise=0%, avg over K values]")

            task_errors = []
            for task_name, task_data in sorted(results["tasks"].items()):
                combos = get_noise0_combos(task_data)
                if not combos:
                    continue

                # Average D_recovered across noise=0 combos
                recoveries = []
                for combo_key in combos:
                    rec = get_recovery_at_step(task_data, combo_key, "maml", step_idx)
                    if rec:
                        for coeff_name, vals in rec.items():
                            recoveries.append(vals)

                if recoveries:
                    avg_rec = np.mean([r["recovered"] for r in recoveries])
                    avg_err = np.mean([r["error_pct"] for r in recoveries
                                       if r["error_pct"] is not None])
                    true_val = recoveries[0]["true"]
                    task_errors.append(avg_err)

                    # For cheat: show individual w_uxx, w_uyy from weight NPZ
                    weight_suffix = ""
                    if model_type == "cheat":
                        w_uxx_vals = []
                        w_uyy_vals = []
                        for combo_key in combos:
                            w_traj = load_weight_trajectory(
                                model_dir, task_name, combo_key, "maml"
                            )
                            if w_traj is not None and step_idx < len(w_traj):
                                w_uxx_vals.append(w_traj[step_idx][3])
                                w_uyy_vals.append(w_traj[step_idx][4])
                        if w_uxx_vals:
                            avg_uxx = np.mean(w_uxx_vals)
                            avg_uyy = np.mean(w_uyy_vals)
                            weight_suffix = f"  w_uxx={avg_uxx:.4f}  w_uyy={avg_uyy:.4f}"

                    lines.append(
                        f"    {task_name:30s}  D_true={true_val:.4f}  "
                        f"D_rec={avg_rec:.4f}  err={avg_err:.1f}%{weight_suffix}"
                    )

            if task_errors:
                mean_err = np.mean(task_errors)
                std_err = np.std(task_errors)
                lines.append(f"    MEAN={mean_err:.1f}%  STD={std_err:.1f}%")
                collector.append(mean_err)
            lines.append("")

        # Cheat-only sections
        if model_type == "cheat":
            theta_star_w = load_theta_star_weights(model_dir)

            # Collect weight trajectories across tasks for split analysis
            split_data = {"theta_star": [], "designed": [], "last": []}
            nonrhs_data = {"theta_star": [], "last": []}

            if theta_star_w is not None:
                split_data["theta_star"].append(
                    (theta_star_w[3], theta_star_w[4])
                )
                nonrhs_data["theta_star"].append(theta_star_w[:3])

            for task_name in sorted(results["tasks"].keys()):
                combos = get_noise0_combos(results["tasks"][task_name])
                for combo_key in combos:
                    w_traj = load_weight_trajectory(
                        model_dir, task_name, combo_key, "maml"
                    )
                    if w_traj is not None:
                        if designed_idx < len(w_traj):
                            w_d = w_traj[designed_idx]
                            split_data["designed"].append((w_d[3], w_d[4]))
                        if last_idx < len(w_traj):
                            w_l = w_traj[last_idx]
                            split_data["last"].append((w_l[3], w_l[4]))
                            nonrhs_data["last"].append(w_l[:3])

            # w_uxx / w_uyy split
            lines.append(f"  [cheat] w_uxx / w_uyy split (noise=0%, avg over K values, tasks):")
            for label, pairs in split_data.items():
                if pairs:
                    avg_uxx = np.mean([p[0] for p in pairs])
                    avg_uyy = np.mean([p[1] for p in pairs])
                    gap = abs(avg_uxx - avg_uyy)
                    lines.append(
                        f"    {label:12s}: w_uxx={avg_uxx:.4f}  w_uyy={avg_uyy:.4f}  gap={gap:.4f}"
                    )
                    if label == "theta_star":
                        all_split_theta_star.append(gap)
                    elif label == "designed":
                        all_split_designed.append(gap)
                    elif label == "last":
                        all_split_last.append(gap)
            lines.append("")

            # Non-RHS weights
            lines.append(f"  [cheat] Non-RHS weights (noise=0%, avg over K values, tasks):")
            for label, arrs in nonrhs_data.items():
                if arrs:
                    avg = np.mean(arrs, axis=0)
                    lines.append(
                        f"    {label:12s}: w_u={avg[0]:+.4f}  w_ux={avg[1]:+.4f}  w_uy={avg[2]:+.4f}"
                    )
                    mag = np.sum(np.abs(avg))
                    if label == "theta_star":
                        all_nonrhs_theta_star.append(mag)
                    elif label == "last":
                        all_nonrhs_last.append(mag)
            lines.append("")
        lines.append("")

    # Variant summary
    lines.append("=" * 60)
    lines.append(f"VARIANT SUMMARY: {variant_name}")
    lines.append("=" * 60)

    if model_type == "cheat" and theta_star is not None:
        d_avg = (theta_star[3] + theta_star[4]) / 2
        lines.append(f"θ* D: (w_uxx + w_uyy)/2 = {d_avg:.4f}")
    lines.append("")

    for label, collector in [
        ("designed step", all_designed_means),
        ("last step", all_last_means),
    ]:
        if collector:
            lines.append(f"Recovery ({label}):")
            lines.append(f"  MEAN of means = {np.mean(collector):.1f}%")
            lines.append(f"  STD of means  = {np.std(collector):.1f}%")
            lines.append("")

    if model_type == "cheat":
        lines.append("[cheat] Mean split gap:")
        for label, vals in [
            ("θ*", all_split_theta_star),
            ("designed step", all_split_designed),
            ("last step", all_split_last),
        ]:
            if vals:
                lines.append(f"  {label:15s}: {np.mean(vals):.4f}")
        lines.append("")

        lines.append("[cheat] Non-RHS magnitude (|w_u|+|w_ux|+|w_uy|):")
        for label, vals in [
            ("θ*", all_nonrhs_theta_star),
            ("last step", all_nonrhs_last),
        ]:
            if vals:
                lines.append(f"  {label:15s}: {np.mean(vals):.4f}")
        lines.append("")

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"  Written: {out_path}")

    # Return summary for cross-variant
    summary = {
        "variant": variant_name,
        "theta_star_d": (theta_star[3] + theta_star[4]) / 2 if (model_type == "cheat" and theta_star is not None) else None,
        "designed_mean": np.mean(all_designed_means) if all_designed_means else None,
        "designed_std": np.std(all_designed_means) if all_designed_means else None,
        "last_mean": np.mean(all_last_means) if all_last_means else None,
        "last_std": np.std(all_last_means) if all_last_means else None,
    }
    return summary


def write_cross_variant_log(summaries: list[dict], out_path: Path):
    """Write cross-variant summary log."""
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-VARIANT SUMMARY (noise=0% only)")
    lines.append("=" * 80)
    lines.append("")

    header = f"{'Variant':<35s} {'θ* D':>8s}   {'Designed err%':>14s}   {'Last step err%':>14s}"
    subhdr = f"{'':35s} {'':>8s}   {'mean':>7s} {'std':>6s}   {'mean':>7s} {'std':>6s}"
    lines.append(header)
    lines.append(subhdr)
    lines.append("─" * 80)

    all_theta_d = []
    all_designed = []
    all_last = []

    for s in summaries:
        td = f"{s['theta_star_d']:.3f}" if s["theta_star_d"] is not None else "N/A"
        dm = f"{s['designed_mean']:.1f}" if s["designed_mean"] is not None else "N/A"
        ds = f"{s['designed_std']:.1f}" if s["designed_std"] is not None else "N/A"
        lm = f"{s['last_mean']:.1f}" if s["last_mean"] is not None else "N/A"
        ls = f"{s['last_std']:.1f}" if s["last_std"] is not None else "N/A"

        lines.append(f"{s['variant']:<35s} {td:>8s}   {dm:>7s} {ds:>6s}   {lm:>7s} {ls:>6s}")

        if s["theta_star_d"] is not None:
            all_theta_d.append(s["theta_star_d"])
        if s["designed_mean"] is not None:
            all_designed.append(s["designed_mean"])
        if s["last_mean"] is not None:
            all_last.append(s["last_mean"])

    lines.append("─" * 80)

    td_mean = f"{np.mean(all_theta_d):.3f}" if all_theta_d else "N/A"
    td_std = f"{np.std(all_theta_d):.3f}" if all_theta_d else ""
    dm_mean = f"{np.mean(all_designed):.1f}" if all_designed else "N/A"
    dm_std = f"{np.std(all_designed):.1f}" if all_designed else "N/A"
    lm_mean = f"{np.mean(all_last):.1f}" if all_last else "N/A"
    lm_std = f"{np.std(all_last):.1f}" if all_last else "N/A"

    lines.append(f"{'ACROSS VARIANTS (mean)':<35s} {td_mean:>8s}   {dm_mean:>7s} {dm_std:>6s}   {lm_mean:>7s} {lm_std:>6s}")
    if all_theta_d:
        lines.append(f"{'ACROSS VARIANTS (std)':<35s} {td_std:>8s}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"  Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate cheat experiment reports")
    parser.add_argument("--model-type", choices=["cheat", "mlp"], required=True)
    parser.add_argument("--variants", nargs="+", type=Path,
                        help="Paths to variant heat dirs (e.g., data/models/cheat/heat)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover cheat variants in data/models/")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if args.auto:
        base = Path("data/models")
        variant_dirs = sorted([
            d / "heat" for d in base.iterdir()
            if d.is_dir() and d.name.startswith("cheat") and (d / "heat").is_dir()
        ])
    elif args.variants:
        variant_dirs = args.variants
    else:
        print("Specify --variants or --auto")
        sys.exit(1)

    print(f"Found {len(variant_dirs)} variants:")
    for v in variant_dirs:
        print(f"  {v}")
    print()

    summaries = []
    for variant_dir in variant_dirs:
        variant_name = variant_dir.parent.name
        print(f"Processing {variant_name}...")
        out_path = args.output_dir / f"{variant_name}.log"
        summary = write_per_variant_log(variant_dir, args.model_type, out_path)
        if summary is not None:
            summaries.append(summary)

    if summaries:
        print(f"\nWriting cross-variant summary...")
        write_cross_variant_log(summaries, args.output_dir / "cheat_cross_variant.log")

    print("\nDone.")


if __name__ == "__main__":
    main()
