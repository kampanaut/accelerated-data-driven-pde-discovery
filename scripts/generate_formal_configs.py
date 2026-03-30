"""Generate formal experiment configs.

Grid per variant: 2 inner_steps × 2 k_shot × 4 loss_modes × 3 PDEs = 48
Variants: formal (standard MLP), zeroed (masked non-RHS features), cheat (linear model)
"""

import yaml
from collections import namedtuple
from pathlib import Path
from typing import Callable, Optional

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
Variant = namedtuple("Variant", [
    "label", "configs_dir", "models_dir", "zero_non_rhs", "layers_fn",
    "log_weights", "weight_init", "overrides",
])

def _cheat_layers(pde: PDE) -> list[dict]:
    """Linear model: input → output with no bias, no hidden layers."""
    return [{"input": pde.input_dim}, {"output": pde.output_dim, "bias": False}]

# Finn-style LR overrides: inner_lr=0.01, fine_tune_lr=0.01, max_steps=50
_FINN_LR = {
    "training": {"inner_lr": 0.01},
    "evaluation": {
        "fine_tune_lr": 0.01,
    },
}

VARIANTS = [
    Variant("formal",              "configs/formal",              "data/models/formal",              False, None,          False, None,       None),
    Variant("zeroed",              "configs/zeroed",              "data/models/zeroed",              True,  None,          False, None,       None),
    Variant("cheat",               "configs/cheat",               "data/models/cheat",               False, _cheat_layers, True,  None,       None),
    Variant("cheat_zeroed",        "configs/cheat_zeroed",        "data/models/cheat_zeroed",        True,  _cheat_layers, True,  None,       None),
    Variant("cheat_zero_init",     "configs/cheat_zero_init",     "data/models/cheat_zero_init",     False, _cheat_layers, True,  "zeros",    None),
    Variant("cheat_zeroed_zinit",  "configs/cheat_zeroed_zinit",  "data/models/cheat_zeroed_zinit",  True,  _cheat_layers, True,  "zeros",    None),
    Variant("cheat_expected",      "configs/cheat_expected",      "data/models/cheat_expected",      False, _cheat_layers, True,  "expected", None),
    Variant("cheat_zeroed_expect", "configs/cheat_zeroed_expect", "data/models/cheat_zeroed_expect", True,  _cheat_layers, True,  "expected", None),
    # Finn-style LR (inner_lr=0.01, fine_tune_lr=0.01, max_eval_steps=50)
    Variant("cheat_finn",               "configs/cheat_finn",               "data/models/cheat_finn",               False, _cheat_layers, True,  None,       _FINN_LR),
    Variant("cheat_zeroed_finn",        "configs/cheat_zeroed_finn",        "data/models/cheat_zeroed_finn",        True,  _cheat_layers, True,  None,       _FINN_LR),
    Variant("cheat_expected_finn",      "configs/cheat_expected_finn",      "data/models/cheat_expected_finn",      False, _cheat_layers, True,  "expected", _FINN_LR),
    Variant("cheat_zeroed_expect_finn", "configs/cheat_zeroed_expect_finn", "data/models/cheat_zeroed_expect_finn", True,  _cheat_layers, True,  "expected", _FINN_LR),
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
    layers_fn: Optional[Callable] = None,
    log_weights: bool = False,
    weight_init: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Build the full config dict for one experiment."""
    flags = loss_mode_flags(loss_mode)
    name = f"{pde.name}-{exp_number}-{inner_steps}step-k{k_shot}-{loss_mode}"
    query_size = 1600 if k_shot == 800 else 2000

    # Model architecture: layers (new) or old fields — inserted here to
    # preserve key ordering in YAML (model fields come after log_interval)
    if layers_fn is not None:
        model_fields: dict = {"layers": layers_fn(pde)}
    else:
        model_fields = {
            "hidden_dims": [100, 100],
            "activation": "silu",
            "input_dim": pde.input_dim,
            "output_dim": pde.output_dim,
        }

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
        **model_fields,
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

    if weight_init is not None:
        training["weight_init"] = weight_init

    if flags["spectral"]:
        training["spectral_loss"] = {"enabled": True}

    if flags["metal"]:
        training["metal"] = {"enabled": True, "hidden_dim": 64}

    config = {
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
            "k_values": [k_shot],
            "noise_levels": [0.0, 0.01, 0.05, 0.10],
            "fine_tune_lr": 0.00001,
            "max_steps": 50,
            "deriv_threshold": 0.0005,
            "fixed_steps": [0] + sorted(set([inner_steps, 5, 10, 25, 50])),
            "holdout_size": 5000,
            "log_weights": log_weights,
        },
        "visualization": {
            "dpi": 300,
            "only": f"scatter[0,{inner_steps},50],best-combo",
        },
    }

    # Apply overrides (e.g., Finn-style LR)
    if overrides:
        for section, values in overrides.items():
            if section in config:
                config[section].update(values)

    return config


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


def build_scatter_groups(specs: list[ExperimentSpec]) -> dict[tuple, list[str]]:
    """Group experiment names by (pde_name, loss_mode, inner_steps) for scatter comparison."""
    groups: dict[tuple, list[str]] = {}
    for spec in specs:
        groups.setdefault((spec.pde.name, spec.loss_mode, spec.inner_steps), []).append(spec.name)
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
                layers_fn=variant.layers_fn,
                log_weights=variant.log_weights,
                weight_init=variant.weight_init,
                overrides=variant.overrides,
            )
            group = scatter_groups[(spec.pde.name, spec.loss_mode, spec.inner_steps)]
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


# ── Grid Part II: MAML++ experiments ────────────────────────────────────
# Grid 1: Cheat convergence (linear model, 8 configs)
# Grid 2: MLP formal (silu + sin activations, 16 configs)
# Total: 24 configs, split into 4 bash scripts of 6 each

_GRID2_LOSS_MODES = ["baseline", "metal", "maml++", "metal+maml++"]
_GRID2_K_SHOTS = [10, 800]

_MAMLPP_FLAGS = {
    "msl_enabled": True,
    "da_enabled": True,
    "da_threshold": 23000,
    "lslr_enabled": True,
    "warmup_iterations": 0,
}

_GRID2_SHARED_TRAINING = {
    "inner_lr": 0.01,
    "outer_lr": 0.001,
    "inner_steps": 5,
    "meta_batch_size": 25,
    "max_iterations": 70000,
    "patience": 0,
    "checkpoint_interval": 500,
    "log_interval": 10,
    "first_order": False,
    "warmup_iterations": 0,
    "use_scheduler": True,
    "scheduler_type": "cosine",
    "min_lr": 0.000001,
    "max_grad_norm": 100.0,
    "zero_non_rhs_features": True,
    "loss_function": "normalized_mse",
}

_GRID2_SHARED_EVAL = {
    "k_values": None,  # filled per-config from k_shot
    "noise_levels": [0.0],
    "fine_tune_lr": 0.01,
    "max_steps": 50,
    "deriv_threshold": 0.0005,
    "fixed_steps": [0, 1, 5, 10, 25, 50],
    "holdout_size": 5000,
}

HEAT = PDES[1]  # heat PDE


def _grid2_loss_flags(mode: str) -> dict:
    """Return training config fragment for a Grid 2 loss mode."""
    flags: dict = {}
    if mode in ("metal", "metal+maml++"):
        flags["metal"] = {"enabled": True, "hidden_dim": 64}
    if mode in ("maml++", "metal+maml++"):
        flags.update(_MAMLPP_FLAGS)
    return flags


def _grid2_config(
    exp_num: int,
    grid_label: str,
    name: str,
    k_shot: int,
    loss_mode: str,
    models_dir: str,
    layers: dict,
    log_weights: bool = False,
) -> dict:
    """Build one Grid 2 config dict."""
    loss_flags = _grid2_loss_flags(loss_mode)

    # Separate metal dict from training-level flags
    metal_cfg = loss_flags.pop("metal", None)

    training = {
        **_GRID2_SHARED_TRAINING,
        **layers,
        "k_shot": k_shot,
        "query_size": 1600 if k_shot == 800 else 2000,
        **loss_flags,
    }

    if metal_cfg is not None:
        training["metal"] = metal_cfg

    eval_cfg = {
        **_GRID2_SHARED_EVAL,
        "k_values": [k_shot],
        "log_weights": log_weights,
    }

    return {
        "experiment": {
            "name": name,
            "pde_type": "heat",
            "seed": 42,
            "device": "cuda",
        },
        "output": {
            "base_dir": models_dir,
        },
        "data": {
            "meta_train_dir": HEAT.train_dir,
            "meta_val_dir": HEAT.val_dir,
            "meta_test_dir": HEAT.test_dir,
        },
        "training": training,
        "evaluation": eval_cfg,
        "visualization": {
            "dpi": 300,
            "only": "scatter[0,5,50],best-combo",
        },
    }


def generate_grid_pt2():
    """Generate cheat2 + mlp configs and 4 runner scripts."""
    cheat_dir = ROOT_DIR / "configs" / "cheat2"
    mlp_dir = ROOT_DIR / "configs" / "mlp"
    cheat_dir.mkdir(parents=True, exist_ok=True)
    mlp_dir.mkdir(parents=True, exist_ok=True)

    all_configs: list[tuple[int, str]] = []  # (exp_num, rel_path)
    exp_num = 0

    print(f"\n{'=' * 60}")
    print("Cheat2: convergence query (linear model)")
    print(f"{'=' * 60}")

    cheat_layers = {"layers": _cheat_layers(HEAT)}

    for loss_mode in _GRID2_LOSS_MODES:
        for k_shot in _GRID2_K_SHOTS:
            exp_num += 1
            name = f"cheat-{exp_num}-5step-k{k_shot}-{loss_mode}"
            config = _grid2_config(
                exp_num, "cheat", name, k_shot, loss_mode,
                "data/models/cheat2", cheat_layers, log_weights=True,
            )
            path = cheat_dir / f"{name}.yaml"
            rel_path = f"configs/cheat2/{name}.yaml"
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            all_configs.append((exp_num, rel_path))
            print(f"  [{exp_num:2d}] {name}")

    print(f"\n{'=' * 60}")
    print("MLP: formal re-run (silu + sin)")
    print(f"{'=' * 60}")

    for activation in ["silu", "sin"]:
        mlp_layers = {
            "hidden_dims": [100, 100],
            "activation": activation,
            "input_dim": HEAT.input_dim,
            "output_dim": HEAT.output_dim,
        }
        for loss_mode in _GRID2_LOSS_MODES:
            for k_shot in _GRID2_K_SHOTS:
                exp_num += 1
                name = f"heat-{exp_num}-5step-k{k_shot}-{loss_mode}-{activation}"
                config = _grid2_config(
                    exp_num, "mlp", name, k_shot, loss_mode,
                    "data/models/mlp", mlp_layers,
                )
                path = mlp_dir / f"{name}.yaml"
                rel_path = f"configs/mlp/{name}.yaml"
                with open(path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                all_configs.append((exp_num, rel_path))
                print(f"  [{exp_num:2d}] {name}")

    print(f"\n  Generated {len(all_configs)} configs total")

    # ── Split into 4 runner scripts of 6 each ──────────────────────
    total = len(all_configs)
    chunk_size = 6
    for script_idx in range(4):
        start = script_idx * chunk_size
        chunk = all_configs[start : start + chunk_size]
        _write_phased_script(
            SCRIPT_DIR / f"run_grid2_{script_idx + 1}.sh", chunk, total,
        )

    _write_phased_script(
        SCRIPT_DIR / "run_grid2_all.sh", all_configs, total,
    )


if __name__ == "__main__":
    import sys
    if "--grid2" in sys.argv:
        generate_grid_pt2()
    else:
        main()
