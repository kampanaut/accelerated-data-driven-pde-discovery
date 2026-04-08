"""Generate experiment configs from variant definitions.

Each variant is a dict of config fields. Values can be:
- Scalar: fixed for all configs
- List of scalars: axis (cartesian product)
- List of (name, dict) tuples: preset axis (dict merges into config, name goes in filename)

A default variant provides baseline values. Named variants override specific fields.
Filenames include only fields that differ from default, using FIELD_LABELS for short names.
"""

import itertools
import yaml
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config import (
    ExperimentConfig, ExperimentSection, OutputSection, DataSection,
    TrainingSection, EvaluationSection, VisualizationSection,
    MetalSection, SpectralLossSection,
)

# ── PDE definitions ──────────────────────────────────────────────────────
PDE = namedtuple("PDE", ["name", "pde_type", "train_dir", "val_dir", "test_dir",
                          "input_dim", "output_dim"])

PDE_REGISTRY = {
    "br": PDE("br", "br",
              "data/datasets/br_train-2", "data/datasets/br_val-2", "data/datasets/br_test-2",
              10, 2),
    "heat": PDE("heat", "heat",
                "data/datasets/heat_train-1", "data/datasets/heat_val-1", "data/datasets/heat_test-1",
                5, 1),
    "nl_heat": PDE("nl_heat", "nl_heat",
                   "data/datasets/nl_heat_train-1", "data/datasets/nl_heat_val-1", "data/datasets/nl_heat_test-1",
                   5, 1),
}

# Sobol variants — same PDE types, different training data
PDE_REGISTRY_SOBOL = {
    "heat": PDE("heat", "heat",
                "data/datasets/heat_train-2", "data/datasets/heat_val-1", "data/datasets/heat_test-1",
                5, 1),
    "nl_heat": PDE("nl_heat", "nl_heat",
                   "data/datasets/nl_heat_train-2", "data/datasets/nl_heat_val-1", "data/datasets/nl_heat_test-1",
                   5, 1),
}

# ── Field labels for filenames ───────────────────────────────────────────
# Only fields that can appear as axes need labels.
# If a field varies from default and has no label here, the generator errors.
FIELD_LABELS = {
    "k_shot": "k",
    "inner_steps": "step",
    "activation": "",
    "inner_lr": "lr",
    "pde_type": "",
}

# Fields where the label is a suffix (e.g. "5step") instead of prefix (e.g. "k10")
FIELD_LABEL_AS_SUFFIX: set[str] = {"inner_steps"}

# Fields always included in filename, even when matching default
FLAG_ALWAYS: set[str] = {"pde_type"}

# Ordering of fields in filename display. Axes not in this list → error.
FILENAME_ORDER: list[str] = ["pde_type", "inner_steps", "k_shot", "loss_preset", "activation"]

# Iteration order for cartesian product (controls numbering).
AXIS_ARRANGEMENT: list[str] = ["pde_type", "inner_steps", "loss_preset", "k_shot", "activation"]

# ── Helpers ──────────────────────────────────────────────────────────────


class Axis:
    """Marks a list of values as a grid axis (cartesian product)."""
    def __init__(self, values: list):
        self.values = values


class Preset:
    """Marks a list of (name, dict) tuples as a preset axis."""
    def __init__(self, entries: list[tuple[str, dict]]):
        self.entries = entries


def _cheat_layers(input_dim: int, output_dim: int) -> list[dict]:
    """Linear model: input → output with no bias, no hidden layers."""
    return [{"input": input_dim}, {"output": output_dim, "bias": False}]


# ── Preset definitions ───────────────────────────────────────────────────

GRID2_LOSS_PRESETS = Preset([
    ("baseline", {}),
    ("metal", {
        "metal": {"enabled": True, "hidden_dim": 0},
    }),
    ("mamlpp", {
        "msl_enabled": True, "da_enabled": True, "da_threshold": 5000, "lslr_enabled": True,
        "use_scheduler": True, "scheduler_type": "cosine", "fine_tune_lr": 0.0001,
    }),
    ("mamlpp+metal", {
        "metal": {"enabled": True, "hidden_dim": 0},
        "msl_enabled": True, "da_enabled": True, "da_threshold": 5000, "lslr_enabled": True,
        "use_scheduler": True, "scheduler_type": "cosine", "fine_tune_lr": 0.0001,
    }),
    ("imaml-lbfgs", {
        "activation": "sin",  # override axis — faithful Raissi setup
        "k_shot": 800,        # override axis — K=10 not relevant for iMAML
        "imaml": {"enabled": True, "lam": 2.0, "lam_lr": 0.001, "cg_steps": 5,
                   "cg_damping": 1.0, "inner_optimizer": "lbfgs"},
    }),
    ("imaml-lbfgs+metal", {
        "activation": "sin",
        "k_shot": 800,
        "metal": {"enabled": True, "hidden_dim": 0},
        "imaml": {"enabled": True, "lam": 2.0, "lam_lr": 0.001, "cg_steps": 5,
                   "cg_damping": 1.0, "inner_optimizer": "lbfgs"},
    }),
])

# ── Default variant ──────────────────────────────────────────────────────
# All variants inherit from this. Override only what differs.

DEFAULT = {
    # PDE
    "pde_type": "heat",
    "k_shot": 1000,

    # Model (Raissi-faithful)
    "hidden_dims": [100, 100],
    "activation": "sin",
    "zero_non_rhs_features": True,

    # Training (iMAML reference defaults)
    "inner_lr": 0.01,
    "outer_lr": 0.01,
    "adam_betas": [0.9, 0.999],
    "inner_steps": 16,
    "meta_batch_size": 25,
    "max_iterations": 1000,
    "patience": 0,
    "checkpoint_interval": 100,
    "log_interval": 20,
    "first_order": False,
    "warmup_iterations": 0,
    "use_scheduler": False,
    "scheduler_type": "cosine",
    "T_0": 200,
    "T_mult": 2,
    "min_lr": 0.00001,
    "max_grad_norm": 0.0,
    "loss_function": "normalized_mse",

    # MAML++ defaults (off — legacy)
    "msl_enabled": False,
    "da_enabled": False,
    "da_threshold": 5000,
    "lslr_enabled": False,

    # MeTAL default (off)
    "metal": {"enabled": False},

    # iMAML default (on — L-BFGS, reference defaults)
    "imaml": {"enabled": True, "lam": 1.0, "cg_steps": 5,
              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
              "proximal_every_step": False},

    # Spectral loss default (off)
    "spectral_loss": {"enabled": False},
    "spectral_loss_mode_size": 32,

    # Evaluation
    "fine_tune_lr": 0.01,
    "max_eval_steps": 16,
    "noise_levels": [0.0],
    "holdout_size": 50000,
    "query_size": 10000,

    # Misc
    "log_weights": False,
    "seed": 42,
    "device": "cuda",
}

# ── Old grid overrides (diffs from DEFAULT) ──────────────────────────────

OLD_LOSS_PRESETS = Preset([
    ("baseline", {}),
    ("metal", {"metal": {"enabled": True, "hidden_dim": 0}}),
    ("spectral", {"spectral_loss": {"enabled": True, "mode_size": 32}}),
    ("metal-spectral", {
        "metal": {"enabled": True, "hidden_dim": 0},
        "spectral_loss": {"enabled": True, "mode_size": 32},
    }),
])

_OLD_BASE = {
    "pde_type": Axis(["br", "heat", "nl_heat"]),
    "inner_steps": Axis([1, 5]),
    "loss_preset": OLD_LOSS_PRESETS,
    "inner_lr": 0.0001,
    "meta_batch_size": 8,
    "max_iterations": 500,
    "patience": 300,
    "checkpoint_interval": 0,
    "zero_non_rhs_features": False,
    "noise_levels": [0.0, 0.01, 0.05, 0.10],
    "fine_tune_lr": 0.00001,
}

_CHEAT_LAYERS = lambda pde: _cheat_layers(pde.input_dim, pde.output_dim)
_FINN_OVERRIDES = {"inner_lr": 0.01, "fine_tune_lr": 0.01}

# ── Variant definitions ──────────────────────────────────────────────────

VariantMeta = namedtuple("VariantMeta", ["label", "configs_dir", "models_dir"])

FINALS_PRESETS = Preset([
    ("sin", {"activation": "sin"}),
    ("silu", {"activation": "silu"}),
    ("silu-cosine", {
        "activation": "silu",
        "use_scheduler": True, "scheduler_type": "cosine",
    }),
    ("silu-lam05", {
        "activation": "silu",
        "imaml": {"enabled": True, "lam": 0.5, "cg_steps": 5,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
    }),
    ("silu-lam05-cosine", {
        "activation": "silu",
        "imaml": {"enabled": True, "lam": 0.5, "cg_steps": 5,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "cosine",
        "min_lr": 1e-7,
    }),
    ("silu-lam05-3layers", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "imaml": {"enabled": True, "lam": 0.5, "cg_steps": 5,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
    }),
    ("silu-lam05-cg10", {
        "activation": "silu",
        "imaml": {"enabled": True, "lam": 0.5, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
    }),
    ("silu-lam05-3layers-cg10-cosine", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "imaml": {"enabled": True, "lam": 0.5, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "cosine",
        "min_lr": 1e-7,
    }),
    ("silu-lam01-cosine", {
        "activation": "silu",
        "imaml": {"enabled": True, "lam": 0.1, "cg_steps": 5,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "cosine",
        "min_lr": 1e-7,
    }),
    ("silu-lam01-3layers-cg10-cosine", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "imaml": {"enabled": True, "lam": 0.1, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "cosine",
        "min_lr": 1e-7,
    }),
    ("silu-lam05-3layers-cg5-cosine", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "imaml": {"enabled": True, "lam": 0.5, "cg_steps": 5,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "cosine",
        "min_lr": 1e-7,
    }),
    ("silu-lam005-3layers-cg10-cosine-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "cosine",
        "min_lr": 1e-7,
    }),
    ("silu-lam01-3layers-cg10-cosine-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.1, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "cosine",
        "min_lr": 1e-7,
    }),
    ("silu-lam005-3layers-cg10-exp", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "exponential",
        "min_lr": 1e-7,
    }),
    ("silu-lam01-3layers-cg10-exp", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "imaml": {"enabled": True, "lam": 0.1, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "exponential",
        "min_lr": 1e-7,
    }),
    ("silu-lam005-3layers-cg10-exp-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "exponential",
        "min_lr": 1e-7,
    }),
    ("silu-lam01-3layers-cg10-exp-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.1, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "exponential",
        "min_lr": 1e-7,
    }),
    ("silu-lam005-3layers-cg10-poly3-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "polynomial",
        "min_lr": 1e-7,
    }),
    ("silu-lam01-3layers-cg10-poly3-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.1, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "polynomial",
        "min_lr": 1e-7,
    }),
    ("silu-lam005-3layers-cg10-plateau-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "plateau",
        "min_lr": 1e-7,
    }),
    ("silu-lam01-3layers-cg10-plateau-2k", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "imaml": {"enabled": True, "lam": 0.1, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "plateau",
        "min_lr": 1e-7,
    }),
    ("silu-lam005-3layers-cg10-poly3-2k-sse", {
        "activation": "silu",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "loss_function": "sse",
        "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "polynomial",
        "min_lr": 1e-7,
    }),
    ("sin-lam005-3layers-cg10-poly3-2k-sse", {
        "activation": "sin",
        "hidden_dims": [100, 100, 100],
        "max_iterations": 2000,
        "loss_function": "sse",
        "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                  "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                  "proximal_every_step": False},
        "use_scheduler": True, "scheduler_type": "polynomial",
        "min_lr": 1e-7,
    }),
])

VARIANTS = [
    # ── Finals: iMAML Heat ablation ──────────────────────────────────
    (
        VariantMeta("finals", "configs/finals", "data/models/finals"),
        {
            "loss_preset": FINALS_PRESETS,
        },
    ),
    # ── The Finals: winner config across PDEs ────────────────────────
    (
        VariantMeta("the-finals", "configs/the-finals", "data/models/the-finals"),
        {
            "pde_type": Axis(["heat", "nl_heat", "br"]),
            "activation": "silu",
            "hidden_dims": [100, 100, 100],
            "loss_preset": Preset([
                ("adam", {
                    "max_iterations": 2000,
                    "meta_batch_size": 25,
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("adam+lbfgs", {
                    "max_iterations": 2100,
                    "meta_batch_size": 8,
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "outer_optimizer": "adam+lbfgs",
                              "outer_lbfgs_after": 2000,
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
            ]),
        },
    ),
    # ── The Finals Sobol: uniform coefficient training sets ─────────────
    (
        VariantMeta("the-finals-sobol", "configs/the-finals-sobol", "data/models/the-finals-sobol"),
        {
            "pde_type": Axis(["heat", "nl_heat"]),
            "pde_registry": PDE_REGISTRY_SOBOL,
            "activation": "silu",
            "loss_preset": Preset([
                ("adam+lbfgs", {
                    "max_iterations": 2100,
                    "meta_batch_size": 8,
                    "hidden_dims": [100, 100, 100],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "outer_optimizer": "adam+lbfgs",
                              "outer_lbfgs_after": 2000,
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("adam-mb25", {
                    "max_iterations": 2000,
                    "meta_batch_size": 25,
                    "hidden_dims": [100, 100, 100],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("adam+lbfgs-5k", {
                    "max_iterations": 5100,
                    "meta_batch_size": 8,
                    "hidden_dims": [100, 100, 100],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "outer_optimizer": "adam+lbfgs",
                              "outer_lbfgs_after": 5000,
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("adam+lbfgs-mb25", {
                    "max_iterations": 2100,
                    "meta_batch_size": 25,
                    "hidden_dims": [100, 100, 100],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "outer_optimizer": "adam+lbfgs",
                              "outer_lbfgs_after": 2000,
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("250x250", {
                    "max_iterations": 2000,
                    "meta_batch_size": 25,
                    "hidden_dims": [250, 250],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("250x250-anil", {
                    "max_iterations": 2000,
                    "meta_batch_size": 25,
                    "hidden_dims": [250, 250],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "anil": True,
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("100x100x100-anil", {
                    "max_iterations": 2000,
                    "meta_batch_size": 25,
                    "hidden_dims": [100, 100, 100],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "anil": True,
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
                ("300x300x300-anil", {
                    "max_iterations": 2000,
                    "meta_batch_size": 25,
                    "hidden_dims": [300, 300, 300],
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "anil": True,
                              "proximal_every_step": False},
                    "use_scheduler": True, "scheduler_type": "polynomial",
                    "poly_power": 3.0,
                    "min_lr": 1e-7,
                }),
            ]),
        },
    ),
    # ── The Finals L-BFGS outer: curvature-aware meta-updates ──────────
    (
        VariantMeta("the-finals-lbfgs-outer", "configs/the-finals-lbfgs-outer", "data/models/the-finals-lbfgs-outer"),
        {
            "pde_type": Axis(["heat", "nl_heat"]),
            "activation": "silu",
            "hidden_dims": [100, 100, 100],
            "max_iterations": 1000,
            "use_scheduler": False,
            "loss_preset": Preset([
                ("mb25", {
                    "meta_batch_size": 25,
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "outer_optimizer": "lbfgs",
                              "proximal_every_step": False},
                }),
                ("full", {
                    "meta_batch_size": 9999,  # full batch — capped by task count
                    "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                              "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                              "outer_optimizer": "lbfgs",
                              "proximal_every_step": False},
                }),
            ]),
        },
    ),
    # ── The Finals 5K: longer training ────────────────────────────────
    (
        VariantMeta("the-finals-5k", "configs/the-finals-5k", "data/models/the-finals-5k"),
        {
            "pde_type": Axis(["heat", "nl_heat"]),
            "activation": "silu",
            "hidden_dims": [100, 100, 100],
            "max_iterations": 5000,
            "meta_batch_size": 25,
            "imaml": {"enabled": True, "lam": 0.005, "cg_steps": 10,
                      "cg_damping": 1.0, "inner_optimizer": "lbfgs",
                      "proximal_every_step": False},
            "use_scheduler": True, "scheduler_type": "polynomial",
            "poly_power": 3.0,
            "min_lr": 1e-7,
        },
    ),
]

# ── Archived variants (MAML era) ─────────────────────────────────────
# These generated configs for the MAML grid (grid1 + grid2).
# Kept for reference. Do not regenerate — results already collected.
#
# _OLD_VARIANTS = [
#     (VariantMeta("formal",              "configs/formal",              "data/models/formal"),              {**_OLD_BASE}),
#     (VariantMeta("zeroed",              "configs/zeroed",              "data/models/zeroed"),              {**_OLD_BASE, "zero_non_rhs_features": True}),
#     (VariantMeta("cheat",               "configs/cheat",               "data/models/cheat"),               {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True}),
#     (VariantMeta("cheat_zeroed",        "configs/cheat_zeroed",        "data/models/cheat_zeroed"),        {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "zero_non_rhs_features": True}),
#     (VariantMeta("cheat_zero_init",     "configs/cheat_zero_init",     "data/models/cheat_zero_init"),     {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "weight_init": "zeros"}),
#     (VariantMeta("cheat_zeroed_zinit",  "configs/cheat_zeroed_zinit",  "data/models/cheat_zeroed_zinit"),  {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "zero_non_rhs_features": True, "weight_init": "zeros"}),
#     (VariantMeta("cheat_expected",      "configs/cheat_expected",      "data/models/cheat_expected"),      {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "weight_init": "expected"}),
#     (VariantMeta("cheat_zeroed_expect", "configs/cheat_zeroed_expect", "data/models/cheat_zeroed_expect"), {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "zero_non_rhs_features": True, "weight_init": "expected"}),
#     (VariantMeta("cheat_finn",               "configs/cheat_finn",               "data/models/cheat_finn"),               {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, **_FINN_OVERRIDES}),
#     (VariantMeta("cheat_zeroed_finn",        "configs/cheat_zeroed_finn",        "data/models/cheat_zeroed_finn"),        {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "zero_non_rhs_features": True, **_FINN_OVERRIDES}),
#     (VariantMeta("cheat_expected_finn",      "configs/cheat_expected_finn",      "data/models/cheat_expected_finn"),      {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "weight_init": "expected", **_FINN_OVERRIDES}),
#     (VariantMeta("cheat_zeroed_expect_finn", "configs/cheat_zeroed_expect_finn", "data/models/cheat_zeroed_expect_finn"), {**_OLD_BASE, "layers": _CHEAT_LAYERS, "log_weights": True, "zero_non_rhs_features": True, "weight_init": "expected", **_FINN_OVERRIDES}),
#     (
#         VariantMeta("cheat2", "configs/cheat2", "data/models/cheat2"),
#         {
#             "loss_preset": GRID2_LOSS_PRESETS,
#             "layers": lambda pde: _cheat_layers(pde.input_dim, pde.output_dim),
#             "log_weights": True,
#             "inner_steps": 10,
#             "max_iterations": 1000,
#         },
#     ),
#     (
#         VariantMeta("mlp", "configs/mlp", "data/models/mlp"),
#         {
#             "loss_preset": GRID2_LOSS_PRESETS,
#             "activation": Axis(["silu", "sin"]),
#         },
#     ),
# ]

# ── Config builder ───────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = Path(__file__).parent.parent


def _merge_variant(default: dict, overrides: dict) -> dict:
    """Merge overrides into default. Returns new dict."""
    merged = deepcopy(default)
    for k, v in overrides.items():
        merged[k] = v
    return merged


def _validate_presets(overrides: dict) -> None:
    """Check that preset-owned keys don't overlap with the variant's non-axis overrides.

    Presets ARE allowed to override axis values (e.g. iMAML preset forces activation=sin).
    """
    preset_keys: set[str] = set()
    non_axis_keys: set[str] = set()
    axis_keys: set[str] = set()

    for key, value in overrides.items():
        if isinstance(value, Preset):
            for _, preset_dict in value.entries:
                preset_keys.update(preset_dict.keys())
        elif isinstance(value, Axis):
            axis_keys.add(key)
        else:
            non_axis_keys.add(key)

    # Only flag overlap with non-axis top-level overrides
    overlap = preset_keys & non_axis_keys
    if overlap:
        raise ValueError(
            f"Fields {overlap} appear both in presets and as top-level variant overrides. "
            f"Remove them from one or the other."
        )


def _collect_axes(merged: dict) -> tuple[list[str], list[list]]:
    """Identify axes (Axis/Preset fields), sorted by AXIS_ARRANGEMENT."""
    unsorted: list[tuple[int, str, list]] = []
    for key, value in merged.items():
        if isinstance(value, Preset):
            order = AXIS_ARRANGEMENT.index(key) if key in AXIS_ARRANGEMENT else len(AXIS_ARRANGEMENT)
            unsorted.append((order, key, value.entries))
        elif isinstance(value, Axis):
            order = AXIS_ARRANGEMENT.index(key) if key in AXIS_ARRANGEMENT else len(AXIS_ARRANGEMENT)
            unsorted.append((order, key, value.values))
    unsorted.sort(key=lambda x: x[0])
    axis_names = [name for _, name, _ in unsorted]
    axis_values = [vals for _, _, vals in unsorted]
    return axis_names, axis_values


def _build_name(
    variant_label: str,
    exp_num: int,
    merged: dict,
    default: dict,
    axis_names: list[str],
    combo: tuple,
) -> str:
    """Build filename from variant label + ordered flags.

    Includes: fields in FLAG_ALWAYS + fields that are axes (vary).
    Order follows FILENAME_ORDER. Axes not in FILENAME_ORDER → error.
    """
    # Map axis names to their current combo values
    axis_vals = dict(zip(axis_names, combo))

    # Validate all axes are in FILENAME_ORDER
    for name in axis_names:
        if name not in FILENAME_ORDER:
            raise ValueError(
                f"Axis '{name}' not in FILENAME_ORDER. Add it before generating."
            )

    parts: list[str] = []
    index_inserted = False

    for field in FILENAME_ORDER:
        is_axis = field in axis_vals
        is_always = field in FLAG_ALWAYS

        if not is_axis and not is_always:
            continue

        if is_axis:
            val = axis_vals[field]
            if isinstance(merged[field], Preset):
                preset_name, _ = val
                parts.append(preset_name)
            else:
                if field not in FIELD_LABELS:
                    raise ValueError(
                        f"Field '{field}' varies but has no entry in FIELD_LABELS."
                    )
                label = FIELD_LABELS[field]
                if field in FIELD_LABEL_AS_SUFFIX:
                    parts.append(f"{val}{label}")
                else:
                    parts.append(f"{label}{val}")
        else:
            # FLAG_ALWAYS, not an axis — use the fixed value
            val = merged[field]
            if field not in FIELD_LABELS:
                raise ValueError(
                    f"Field '{field}' in FLAG_ALWAYS but has no entry in FIELD_LABELS."
                )
            label = FIELD_LABELS[field]
            parts.append(f"{label}{val}")

        # Insert index after the first part (e.g. "heat-1-...")
        if not index_inserted and parts:
            parts.append(str(exp_num))
            index_inserted = True

    return "-".join(parts)


def _flatten_combo(
    merged: dict, axis_names: list[str], combo: tuple
) -> dict[str, Any]:
    """Flatten merged dict + combo into a single dict of resolved values."""
    flat: dict[str, Any] = {}
    for key, value in merged.items():
        if key in axis_names:
            continue
        if callable(value):
            continue
        flat[key] = value

    # First pass: set axis values
    preset_dicts: list[dict] = []
    for axis_name, axis_val in zip(axis_names, combo):
        if isinstance(merged[axis_name], Preset):
            _, preset_dict = axis_val
            preset_dicts.append(preset_dict)
        else:
            flat[axis_name] = axis_val

    # Second pass: preset overrides win over axis values
    for preset_dict in preset_dicts:
        flat.update(preset_dict)

    return flat


def _build_config(
    merged: dict,
    axis_names: list[str],
    combo: tuple,
    meta: VariantMeta,
    exp_name: str,
) -> dict:
    """Build YAML config dict via ExperimentConfig (single source of truth)."""
    flat = _flatten_combo(merged, axis_names, combo)

    # Resolve PDE
    registry = flat.get("pde_registry", PDE_REGISTRY)
    pde = registry[flat["pde_type"]]

    # Resolve layers
    if "layers" in merged and callable(merged["layers"]):
        layers_kwargs: dict[str, Any] = {"layers": merged["layers"](pde)}
    else:
        layers_kwargs = {
            "hidden_dims": flat.get("hidden_dims"),
            "activation": flat.get("activation"),
            "input_dim": pde.input_dim,
            "output_dim": pde.output_dim,
        }

    k_shot = flat["k_shot"]
    query_size = flat.get("query_size", 10000)

    # Fixed steps: include inner_steps as designed step
    inner_steps = flat.get("inner_steps", 5)
    max_eval = flat["max_eval_steps"]
    fixed_steps = sorted(set([0, max_eval]))

    # Metal / spectral as section objects
    metal_raw = flat.get("metal", {})
    metal = MetalSection(**(metal_raw if isinstance(metal_raw, dict) else {}))

    spectral_raw = flat.get("spectral_loss", {})
    spectral = SpectralLossSection(**(spectral_raw if isinstance(spectral_raw, dict) else {}))

    imaml_raw = flat.get("imaml", {})
    from src.config import IMAMLSection
    imaml = IMAMLSection(**(imaml_raw if isinstance(imaml_raw, dict) else {}))

    cfg = ExperimentConfig(
        experiment=ExperimentSection(
            name=exp_name,
            pde_type=pde.pde_type,
            seed=flat.get("seed", 42),
            device=flat.get("device", "cuda"),
        ),
        output=OutputSection(
            base_dir=(f"{meta.models_dir}/{pde.name}"
                      if isinstance(merged.get("pde_type"), Axis)
                      else meta.models_dir),
        ),
        data=DataSection(
            meta_train_dir=flat.get("meta_train_dir", pde.train_dir),
            meta_val_dir=pde.val_dir,
            meta_test_dir=pde.test_dir,
        ),
        training=TrainingSection(
            inner_lr=flat["inner_lr"],
            inner_steps=inner_steps,
            outer_lr=flat["outer_lr"],
            adam_betas=flat.get("adam_betas", [0.9, 0.99]),
            meta_batch_size=flat["meta_batch_size"],
            k_shot=k_shot,
            query_size=query_size,
            max_iterations=flat["max_iterations"],
            patience=flat["patience"],
            checkpoint_interval=flat["checkpoint_interval"],
            log_interval=flat["log_interval"],
            first_order=flat["first_order"],
            msl_enabled=flat.get("msl_enabled", False),
            da_enabled=flat.get("da_enabled", False),
            da_threshold=flat.get("da_threshold", 5000),
            lslr_enabled=flat.get("lslr_enabled", False),
            warmup_iterations=flat["warmup_iterations"],
            use_scheduler=flat["use_scheduler"],
            scheduler_type=flat["scheduler_type"],
            T_0=flat.get("T_0", 200),
            T_mult=flat.get("T_mult", 2),
            min_lr=flat["min_lr"],
            loss_function=flat["loss_function"],
            max_grad_norm=flat["max_grad_norm"],
            zero_non_rhs_features=flat["zero_non_rhs_features"],
            weight_init=flat.get("weight_init"),
            metal=metal,
            spectral_loss=spectral,
            imaml=imaml,
            **layers_kwargs,
        ),
        evaluation=EvaluationSection(
            k_values=[k_shot],
            noise_levels=flat.get("noise_levels", [0.0]),
            fine_tune_lr=flat["fine_tune_lr"],
            max_steps=flat.get("max_eval_steps", 50),
            fixed_steps=fixed_steps,
            holdout_size=flat["holdout_size"],
            log_weights=flat.get("log_weights", False),
        ),
        visualization=VisualizationSection(
            dpi=300,
            only=f"scatter[0,1,2,3,4,{inner_steps},{max_eval}],jacobian[0,{inner_steps},{max_eval}],generalization,best-combo",
        ),
    )

    # Round-trip validation
    yaml_dict = cfg.to_yaml_dict()
    ExperimentConfig.from_dict(yaml_dict)

    return yaml_dict


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


# ── Main ─────────────────────────────────────────────────────────────────

def generate_variant(meta: VariantMeta, overrides: dict, default: dict = DEFAULT) -> None:
    """Generate all configs and runner scripts for one variant."""
    _validate_presets(overrides)
    merged = _merge_variant(default, overrides)
    axis_names, axis_values = _collect_axes(merged)

    configs_dir = ROOT_DIR / meta.configs_dir
    configs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"{meta.label}: {meta.configs_dir} → {meta.models_dir}")
    print(f"  Axes: {', '.join(axis_names)}")
    print(f"{'=' * 60}")

    all_entries: list[tuple[int, str]] = []
    # For scatter grouping: group configs that share everything except k_shot
    scatter_groups: dict[tuple, list[str]] = {}
    exp_num = 0
    seen_configs: set[tuple] = set()  # dedup preset-overridden axis combos

    for combo in itertools.product(*axis_values):
        # Flatten to detect preset overrides, build dedup key
        flat = _flatten_combo(merged, axis_names, combo)
        # Dedup key: preset name + overridden axis values
        dedup_parts = []
        for ax, val in zip(axis_names, combo):
            if isinstance(merged[ax], Preset):
                dedup_parts.append(val[0])  # preset name
            elif ax in flat:
                dedup_parts.append(str(flat[ax]))  # use overridden value
            else:
                dedup_parts.append(str(val))
        dedup_key = tuple(dedup_parts)
        if dedup_key in seen_configs:
            continue
        seen_configs.add(dedup_key)

        exp_num += 1
        # Use flat values for name when preset overrides an axis
        name_combo = list(combo)
        for i, ax in enumerate(axis_names):
            if not isinstance(merged[ax], Preset) and ax in flat:
                name_combo[i] = flat[ax]
        name = _build_name(meta.label, exp_num, merged, default, axis_names, tuple(name_combo))
        config = _build_config(merged, axis_names, combo, meta, name)

        # Build scatter group key: all axis values except k_shot
        group_key = tuple(
            (preset_name if isinstance(merged[ax], Preset) else val)
            for ax, val in zip(axis_names, combo)
            if ax != "k_shot"
            for preset_name in ([val[0]] if isinstance(merged[ax], Preset) else [val])
        )
        scatter_groups.setdefault(group_key, []).append(name)

        path = configs_dir / f"{name}.yaml"
        rel_path = f"{meta.configs_dir}/{name}.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        all_entries.append((exp_num, rel_path))
        print(f"  [{exp_num:2d}] {name}")

    # Second pass: set compare_experiments in each config
    for path in configs_dir.glob("*.yaml"):
        with open(path) as f:
            config = yaml.safe_load(f)
        exp_name = config["experiment"]["name"]
        for group in scatter_groups.values():
            if exp_name in group:
                config["visualization"]["compare_experiments"] = group
                break
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    total = exp_num
    print(f"\n  Generated {total} configs in {configs_dir}")

    # Per-PDE scripts — group entries by PDE name from filename
    pde_groups: dict[str, list[tuple[int, str]]] = {}
    for i, rel_path in all_entries:
        pde_name = Path(rel_path).stem.split("-")[0]
        pde_groups.setdefault(pde_name, []).append((i, rel_path))

    for pde_name, pde_entries in pde_groups.items():
        _write_phased_script(
            SCRIPT_DIR / f"run_{meta.label}_{pde_name}.sh", pde_entries, total,
        )

    _write_phased_script(
        SCRIPT_DIR / f"run_{meta.label}_experiments.sh", all_entries, total,
    )

    # Split for parallel execution (4 configs per chunk)
    chunk_size = 4
    pde_list = list(pde_groups.keys())
    pde_label = pde_list[0] if len(pde_list) == 1 else "all"
    for i in range(0, len(all_entries), chunk_size):
        idx = (i // chunk_size) + 1
        chunk = all_entries[i : i + chunk_size]
        _write_phased_script(
            SCRIPT_DIR / f"run_{meta.label}_{pde_label}_{idx}.sh", chunk, total,
        )


def main():
    for meta, overrides in VARIANTS:
        generate_variant(meta, overrides)


if __name__ == "__main__":
    main()
