"""Generate formal experiment configs (with and without zeroed non-RHS features).

Grid per variant: 2 inner_steps × 2 k_shot × 4 loss_modes × 3 PDEs = 48
Total: 48 formal + 48 zeroed = 96 configs
"""

import yaml
from collections import namedtuple
from pathlib import Path

# ── PDE definitions ──────────────────────────────────────────────────────
PDE = namedtuple("PDE", ["name", "pde_type", "train_dir", "val_dir", "test_dir",
                          "input_dim", "output_dim"])

PDES = [
    PDE("br", "br",
        "data/datasets/br_train-2", "data/datasets/br_val-2", "data/datasets/br_test-2",
        10, 2),
    PDE("heat", "heat",
        "data/datasets/heat_train-1", "data/datasets/heat_val-1", "data/datasets/heat_test-1",
        5, 1),
    PDE("nl_heat", "nl_heat",
        "data/datasets/nl_heat_train-1", "data/datasets/nl_heat_val-1", "data/datasets/nl_heat_test-1",
        5, 1),
]

# ── Grid axes ────────────────────────────────────────────────────────────
INNER_STEPS = [1, 5]
K_SHOTS = [800, 10]
LOSS_MODES = ["baseline", "metal", "spectral", "metal-spectral"]

# ── Shared constants ─────────────────────────────────────────────────────
FIXED_STEPS = [0, 1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 800, 1000]

# ── Variant definitions ─────────────────────────────────────────────────
Variant = namedtuple("Variant", ["label", "configs_dir", "models_dir", "zero_non_rhs"])

VARIANTS = [
    Variant("formal",  "configs/formal",  "data/models/formal",  False),
    Variant("zeroed",  "configs/zeroed",  "data/models/zeroed",  True),
]


def loss_mode_flags(mode: str) -> dict:
    """Return the metal/spectral config fragments for a loss mode."""
    metal = mode in ("metal", "metal-spectral")
    spectral = mode in ("spectral", "metal-spectral")
    return {"metal": metal, "spectral": spectral}


def generate_config(
    pde: PDE,
    exp_number: int,
    inner_steps: int,
    k_shot: int,
    loss_mode: str,
    models_dir: str,
    zero_non_rhs_features: bool,
) -> dict:
    """Build the full config dict for one experiment."""
    flags = loss_mode_flags(loss_mode)
    name = f"{pde.name}-{exp_number}-{inner_steps}step-k{k_shot}-{loss_mode}"
    query_size = 1600 if k_shot == 800 else 2000

    training: dict = {
        "inner_lr": 0.0001,
        "outer_lr": 0.001,
        "inner_steps": inner_steps,
        "meta_batch_size": 8,
        "k_shot": k_shot,
        "query_size": query_size,
        "max_iterations": 500,
        "patience": 300,
        "log_interval": 1,
        "hidden_dims": [100, 100],
        "activation": "silu",
        "input_dim": pde.input_dim,
        "output_dim": pde.output_dim,
        "first_order": False,
        "warmup_iterations": 100,
        "use_scheduler": True,
        "scheduler_type": "warm_restarts",
        "T_0": 200,
        "T_mult": 2,
        "min_lr": 0.000001,
        "max_grad_norm": 100.0,
        "zero_non_rhs_features": zero_non_rhs_features,
    }

    if flags["spectral"]:
        training["spectral_loss"] = {"enabled": True}

    if flags["metal"]:
        training["metal"] = {"enabled": True, "hidden_dim": 64}

    return {
        "experiment": {
            "name": name,
            "pde_type": pde.pde_type,
            "seed": 42,
            "device": "cuda",
        },
        "output": {
            "base_dir": f"{models_dir}/{pde.name}",
        },
        "data": {
            "meta_train_dir": pde.train_dir,
            "meta_val_dir": pde.val_dir,
            "meta_test_dir": pde.test_dir,
        },
        "training": training,
        "evaluation": {
            "k_values": [800, 1000, 1600, 3000, 5000],
            "noise_levels": [0.0, 0.01, 0.05, 0.10],
            "fine_tune_lr": 0.00001,
            "max_steps": 1000,
            "deriv_threshold": 0.0005,
            "fixed_steps": FIXED_STEPS,
            "holdout_size": 5000,
        },
        "visualization": {
            "dpi": 300,
            "only": "scatter,jacobian[0,1,25,500,1000],best-combo",
        },
    }


SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = Path(__file__).parent.parent


ExperimentSpec = namedtuple(
    "ExperimentSpec", ["pde", "inner_steps", "k_shot", "loss_mode", "exp_number", "name"]
)


def enumerate_experiments() -> list[ExperimentSpec]:
    """First pass: compute all experiment names and metadata."""
    specs = []
    exp_number = 0
    for pde in PDES:
        for inner_steps in INNER_STEPS:
            for loss_mode in LOSS_MODES:
                for k_shot in K_SHOTS:
                    exp_number += 1
                    name = f"{pde.name}-{exp_number}-{inner_steps}step-k{k_shot}-{loss_mode}"
                    specs.append(ExperimentSpec(pde, inner_steps, k_shot, loss_mode, exp_number, name))
    return specs


def build_scatter_groups(specs: list[ExperimentSpec]) -> dict[tuple[str, str], list[str]]:
    """Group experiment names by (pde_name, loss_mode) for scatter comparison."""
    groups: dict[tuple[str, str], list[str]] = {}
    for spec in specs:
        groups.setdefault((spec.pde.name, spec.loss_mode), []).append(spec.name)
    return groups


def _write_phased_script(path: Path, entries: list[tuple[int, str]], total: int) -> None:
    """Write a script that trains all, then evaluates all, then visualizes all."""
    lines = ["#!/usr/bin/env bash", "set -e", ""]
    lines.append("# === TRAIN ===")
    for i, cfg in entries:
        lines.append(f'echo "=== TRAIN [{i}/{total}] {Path(cfg).stem} ==="')
        lines.append(f"uv run python scripts/train_maml.py --config {cfg}")
        lines.append("")
    lines.append("# === EVALUATE ===")
    for i, cfg in entries:
        lines.append(f'echo "=== EVAL [{i}/{total}] {Path(cfg).stem} ==="')
        lines.append(f"uv run python scripts/evaluate.py --config {cfg}")
        lines.append("")
    lines.append("# === VISUALIZE ===")
    for i, cfg in entries:
        lines.append(f'echo "=== VIS [{i}/{total}] {Path(cfg).stem} ==="')
        lines.append(f"uv run python scripts/visualize.py --config {cfg}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    path.chmod(0o755)
    print(f"  Generated {path}")


def main():
    specs = enumerate_experiments()
    scatter_groups = build_scatter_groups(specs)
    total = len(specs)

    for variant in VARIANTS:
        configs_dir = ROOT_DIR / variant.configs_dir
        configs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Generating {variant.label} configs (zero_non_rhs={variant.zero_non_rhs})")
        print(f"{'=' * 60}")

        config_paths: list[str] = []
        for spec in specs:
            config = generate_config(
                spec.pde, spec.exp_number, spec.inner_steps, spec.k_shot,
                spec.loss_mode, variant.models_dir, variant.zero_non_rhs,
            )
            group = scatter_groups[(spec.pde.name, spec.loss_mode)]
            config["visualization"]["compare_experiments"] = group
            name = config["experiment"]["name"]
            rel_path = f"{variant.configs_dir}/{name}.yaml"
            path = configs_dir / f"{name}.yaml"
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            config_paths.append(rel_path)
            print(f"  [{spec.exp_number:2d}/{total}] {path.name}")

        print(f"\n  Generated {len(specs)} configs in {configs_dir}")

        # Generate bash runner scripts — one per PDE + one combined
        pde_configs: dict[str, list[tuple[int, str]]] = {}
        for i, cfg in enumerate(config_paths, 1):
            pde_name = Path(cfg).stem.split("-")[0]
            pde_configs.setdefault(pde_name, []).append((i, cfg))

        for pde_name, entries in pde_configs.items():
            _write_phased_script(
                SCRIPT_DIR / f"run_{variant.label}_{pde_name}.sh", entries, total,
            )

        all_entries = [(i, cfg) for i, cfg in enumerate(config_paths, 1)]
        _write_phased_script(
            SCRIPT_DIR / f"run_{variant.label}_experiments.sh", all_entries, total,
        )


if __name__ == "__main__":
    main()
