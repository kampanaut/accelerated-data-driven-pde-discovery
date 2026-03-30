"""Unified experiment configuration.

Single source of truth for YAML ↔ dataclass translation.
Scripts use ExperimentConfig.from_yaml(path) instead of manual .get() parsing.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Helpers ──────────────────────────────────────────────────────────────


def _filter_fields(cls: type, d: dict) -> dict:
    """Keep only keys that match dataclass fields. Silently drops unknown keys."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in known}


# ── Section dataclasses ──────────────────────────────────────────────────


@dataclass
class ExperimentSection:
    name: str = ""
    pde_type: str = ""
    seed: int = 42
    device: str = "cuda"


@dataclass
class OutputSection:
    base_dir: str = "data/models"


@dataclass
class DataSection:
    meta_train_dir: str = ""
    meta_val_dir: str = ""
    meta_test_dir: str = ""


@dataclass
class MetalSection:
    enabled: bool = False
    hidden_dim: int = 64


@dataclass
class SpectralLossSection:
    enabled: bool = False
    mode_size: int = 32


@dataclass
class TrainingSection:
    """All training fields as they appear in YAML."""

    # Inner loop
    inner_lr: float = 0.01
    inner_steps: int = 1

    # Outer loop
    outer_lr: float = 0.001
    adam_betas: list = field(default_factory=lambda: [0.9, 0.99])
    meta_batch_size: int = 4

    # Support/query
    k_shot: int = 100
    query_size: int = 1000

    # Training loop
    max_iterations: int = 10000
    patience: int = 50
    checkpoint_interval: int = 0
    log_interval: int = 10

    # First-order
    first_order: bool = False

    # MAML++ (Antoniou et al., ICLR 2019)
    msl_enabled: bool = False
    da_enabled: bool = False
    da_threshold: int = 200
    lslr_enabled: bool = False

    # LR scheduler
    warmup_iterations: int = 0
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    T_0: int = 500
    T_mult: int = 2
    min_lr: float = 1e-6

    # Loss
    loss_function: str = "normalized_mse"

    # Gradient clipping
    max_grad_norm: float = 0.0

    # Feature masking
    zero_non_rhs_features: bool = False

    # Weight initialization (optional: "zeros", "expected", or None for default)
    weight_init: Optional[str] = None

    # Nested sections
    metal: MetalSection = field(default_factory=MetalSection)
    spectral_loss: SpectralLossSection = field(default_factory=SpectralLossSection)

    # Network architecture (XOR: layers OR hidden_dims+activation+input_dim+output_dim)
    hidden_dims: Optional[list] = None
    activation: Optional[str] = None
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    layers: Optional[list] = None
    conv_filters: int = 0
    conv_kernel_size: int = 3


@dataclass
class EvaluationSection:
    k_values: list = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    noise_levels: list = field(default_factory=lambda: [0.0])
    fine_tune_lr: float = 0.01
    max_steps: int = 1000
    deriv_threshold: float = 0.0005
    fixed_steps: list = field(default_factory=lambda: [0, 1, 5, 10, 25, 50])
    holdout_size: int = 5000
    log_weights: bool = False
    zero_non_rhs_features: Optional[bool] = None  # None = inherit from training


@dataclass
class VisualizationSection:
    dpi: int = 300
    only: Optional[str] = None
    compare_experiments: list = field(default_factory=list)


# ── ExperimentConfig ─────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Unified experiment configuration covering all YAML sections."""

    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    output: OutputSection = field(default_factory=OutputSection)
    data: DataSection = field(default_factory=DataSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    visualization: VisualizationSection = field(default_factory=VisualizationSection)

    # ── Parsing ──────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load from YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_yaml_dict(d)

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "ExperimentConfig":
        """Parse raw YAML dict into typed config."""
        experiment = ExperimentSection(**_filter_fields(ExperimentSection, d.get("experiment", {})))
        output = OutputSection(**_filter_fields(OutputSection, d.get("output", {})))
        data = DataSection(**_filter_fields(DataSection, d.get("data", {})))

        # Training: pop nested dicts before spreading flat fields
        train_raw = dict(d.get("training", {}))
        metal = MetalSection(**train_raw.pop("metal", {}))
        spectral = SpectralLossSection(**train_raw.pop("spectral_loss", {}))
        training = TrainingSection(
            **_filter_fields(TrainingSection, train_raw),
            metal=metal,
            spectral_loss=spectral,
        )

        evaluation = EvaluationSection(**_filter_fields(EvaluationSection, d.get("evaluation", {})))
        visualization = VisualizationSection(**_filter_fields(VisualizationSection, d.get("visualization", {})))

        return cls(
            experiment=experiment,
            output=output,
            data=data,
            training=training,
            evaluation=evaluation,
            visualization=visualization,
        )

    # ── Serialization ────────────────────────────────────────────────

    def to_yaml_dict(self) -> dict:
        """Serialize to canonical YAML dict structure."""
        t = self.training

        # Build training dict from flat fields
        train_dict: dict[str, Any] = {}
        # Collect all flat fields (skip nested sections and network fields)
        _nested = {"metal", "spectral_loss"}
        _network = {"hidden_dims", "activation", "input_dim", "output_dim", "layers", "conv_filters", "conv_kernel_size"}

        for f in fields(TrainingSection):
            if f.name in _nested or f.name in _network:
                continue
            val = getattr(t, f.name)
            if val is not None:
                train_dict[f.name] = val

        # Nested sections — only emit if enabled
        if t.metal.enabled:
            train_dict["metal"] = {"enabled": t.metal.enabled, "hidden_dim": t.metal.hidden_dim}
        if t.spectral_loss.enabled:
            train_dict["spectral_loss"] = {"enabled": t.spectral_loss.enabled, "mode_size": t.spectral_loss.mode_size}

        # Network config — preserve whichever format was provided
        if t.layers is not None:
            train_dict["layers"] = t.layers
        else:
            for name in ("hidden_dims", "activation", "input_dim", "output_dim"):
                val = getattr(t, name)
                if val is not None:
                    train_dict[name] = val

        # Conv fields only if non-default
        if t.conv_filters > 0:
            train_dict["conv_filters"] = t.conv_filters
            train_dict["conv_kernel_size"] = t.conv_kernel_size

        # Build evaluation dict
        eval_dict: dict[str, Any] = {}
        for f in fields(EvaluationSection):
            val = getattr(self.evaluation, f.name)
            if val is not None:
                eval_dict[f.name] = val

        # Build visualization dict
        vis_dict: dict[str, Any] = {}
        for f in fields(VisualizationSection):
            val = getattr(self.visualization, f.name)
            if val is not None:
                vis_dict[f.name] = val

        return {
            "experiment": {
                "name": self.experiment.name,
                "pde_type": self.experiment.pde_type,
                "seed": self.experiment.seed,
                "device": self.experiment.device,
            },
            "output": {
                "base_dir": self.output.base_dir,
            },
            "data": {
                "meta_train_dir": self.data.meta_train_dir,
                "meta_val_dir": self.data.meta_val_dir,
                "meta_test_dir": self.data.meta_test_dir,
            },
            "training": train_dict,
            "evaluation": eval_dict,
            "visualization": vis_dict,
        }

    def to_yaml(self, path: Path) -> None:
        """Write to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_yaml_dict(), f, default_flow_style=False, sort_keys=False)

    # ── Conversion to internal configs ───────────────────────────────

    def to_maml_config(self) -> "MAMLConfig":
        """Build MAMLConfig from this config.

        Handles renames, flattening of nested sections, and cross-section fields.
        """
        from .training.maml import MAMLConfig

        t = self.training
        return MAMLConfig(
            inner_lr=t.inner_lr,
            inner_steps=t.inner_steps,
            outer_lr=t.outer_lr,
            adam_betas=tuple(t.adam_betas),
            meta_batch_size=t.meta_batch_size,
            k_shot=t.k_shot,
            query_size=t.query_size,
            max_outer_iterations=t.max_iterations,
            patience=t.patience,
            checkpoint_interval=t.checkpoint_interval,
            log_interval=t.log_interval,
            first_order=t.first_order,
            msl_enabled=t.msl_enabled,
            da_enabled=t.da_enabled,
            da_threshold=t.da_threshold,
            lslr_enabled=t.lslr_enabled,
            device=self.experiment.device,
            seed=self.experiment.seed,
            warmup_iterations=t.warmup_iterations,
            use_scheduler=t.use_scheduler,
            min_lr=t.min_lr,
            scheduler_type=t.scheduler_type,
            T_0=t.T_0,
            T_mult=t.T_mult,
            loss_function=t.loss_function,
            metal_enabled=t.metal.enabled,
            metal_hidden_dim=t.metal.hidden_dim,
            spectral_loss_enabled=t.spectral_loss.enabled,
            spectral_loss_mode_size=t.spectral_loss.mode_size,
            max_grad_norm=t.max_grad_norm,
            zero_non_rhs_features=t.zero_non_rhs_features,
        )

    def to_network_config(self) -> "NetworkConfig":
        """Build NetworkConfig from the network fields in the training section."""
        from .networks.pde_operator_network import NetworkConfig

        t = self.training
        d: dict[str, Any] = {}
        if t.layers is not None:
            d["layers"] = t.layers
        else:
            if t.hidden_dims is not None:
                d["hidden_dims"] = t.hidden_dims
            if t.activation is not None:
                d["activation"] = t.activation
            if t.input_dim is not None:
                d["input_dim"] = t.input_dim
            if t.output_dim is not None:
                d["output_dim"] = t.output_dim
        if t.conv_filters > 0:
            d["conv_filters"] = t.conv_filters
            d["conv_kernel_size"] = t.conv_kernel_size
        return NetworkConfig.from_dict(d)

    # ── Convenience properties ───────────────────────────────────────

    @property
    def eval_zero_non_rhs_features(self) -> bool:
        """Evaluation zero_non_rhs: uses eval override if set, else training value."""
        if self.evaluation.zero_non_rhs_features is not None:
            return self.evaluation.zero_non_rhs_features
        return self.training.zero_non_rhs_features

    @property
    def exp_dir(self) -> Path:
        """Experiment output directory."""
        return Path(self.output.base_dir) / self.experiment.name
