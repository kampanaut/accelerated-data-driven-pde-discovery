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
class SpectralLossSection:
    enabled: bool = False
    mode_size: int = 32


# Backward-compat stub for unpickling old checkpoints that contain a
# MetalSection in their training-section pickle. MeTAL was removed in
# commit 2f494c6; this stub lets torch.load reconstruct old files.
class MetalSection:
    def __init__(self, *args, **kwargs):
        self.enabled = False


@dataclass
class IMAMLSection:
    enabled: bool = False
    lam: float = 1.0
    lam_lr: float = 0.0        # 0 = fixed lambda, >0 = meta-learn lambda
    lam_min: float = 0.0
    cg_steps: int = 5
    cg_damping: float = 1.0
    inner_optimizer: str = "lbfgs"  # "sgd" or "lbfgs"
    outer_optimizer: str = "adam"  # "adam", "lbfgs", or "adam+lbfgs"
    outer_lbfgs_after: int = 0  # for "adam+lbfgs": switch to L-BFGS after this many iters
    anil: bool = False  # True = only adapt last layer during inner loop (ANIL)
    anil_mode: str = "head"  # "head" | "head+scales_all" | "head+scales_last"
    proximal_every_step: bool = False  # True = paper Eq.3, False = reference code (prox at end only)
    slope_recovery_inner: float = 0.0  # weight on S(φ_a) in inner task loss
    slope_recovery_outer: float = 0.0  # weight on S(θ_a) added to outer gradient (bypasses CG)


@dataclass
class TrainingSection:
    """All training fields as they appear in YAML.

    Field order matches the canonical YAML key order (used by to_yaml_dict).
    """

    # Inner/outer loop
    inner_lr: float = 0.01
    outer_lr: float = 0.01
    adam_betas: list = field(default_factory=lambda: [0.9, 0.999])
    inner_steps: int = 16
    meta_batch_size: int = 25

    # Support/query
    k_shot: int = 100
    query_size: int = 1000

    # Training loop
    epochs: int = 1
    max_iterations: int = 1000  # per epoch
    patience: int = 50
    checkpoint_interval: int = 0
    log_interval: int = 20

    # Network architecture — between log_interval and first_order to match old YAML order
    # (XOR: layers OR hidden_dims+activation+input_dim+output_dim)
    hidden_dims: Optional[list] = None
    activation: Optional[str] = None
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    layers: Optional[list] = None
    input_bypass: bool = False
    adaptive_scales: bool = False
    adaptive_scale_n: float = 1.0
    conv_filters: int = 0
    conv_kernel_size: int = 3

    # First-order
    first_order: bool = False

    # MAML++ (Antoniou et al., ICLR 2019)
    msl_enabled: bool = False
    da_enabled: bool = False
    da_threshold: int = 5000
    lslr_enabled: bool = False

    # LR scheduler
    warmup_iterations: int = 0
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    T_0: int = 500
    T_mult: int = 2
    min_lr: float = 1e-5
    poly_power: float = 3.0
    plateau_factor: float = 0.8
    plateau_patience: int = 70
    plateau_cooldown: int = 40
    plateau_threshold: float = 1e-3
    plateau_window: int = 20

    # Loss
    loss_function: str = "normalized_mse"

    # Gradient clipping
    max_grad_norm: float = 0.0

    # Nested sections
    spectral_loss: SpectralLossSection = field(default_factory=SpectralLossSection)
    imaml: IMAMLSection = field(default_factory=IMAMLSection)


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


@dataclass
class VisualizationSection:
    dpi: int = 300
    only: Optional[str] = None
    compare_experiments: list = field(default_factory=list)
    exclude_suffixes_append: list = field(default_factory=list)
    exclude_max_iteration: int = 20


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
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Parse raw YAML dict into typed config."""
        experiment = ExperimentSection(**_filter_fields(ExperimentSection, d.get("experiment", {})))
        output = OutputSection(**_filter_fields(OutputSection, d.get("output", {})))
        data = DataSection(**_filter_fields(DataSection, d.get("data", {})))

        # Training: pop nested dicts before spreading flat fields
        train_raw = dict(d.get("training", {}))
        train_raw.pop("metal", None)  # legacy key, silently discarded
        spectral = SpectralLossSection(**train_raw.pop("spectral_loss", {}))
        imaml = IMAMLSection(**train_raw.pop("imaml", {}))
        training = TrainingSection(
            **_filter_fields(TrainingSection, train_raw),
            spectral_loss=spectral,
            imaml=imaml,
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

        # Build training dict from flat fields in dataclass field order
        train_dict: dict[str, Any] = {}
        # Network fields that are mutually exclusive (layers XOR hidden_dims+friends)
        _layers_format = {"hidden_dims", "activation", "input_dim", "output_dim"}
        _conv = {"conv_filters", "conv_kernel_size"}

        for f in fields(TrainingSection):
            name = f.name
            val = getattr(t, name)

            # Nested sections — emit inline if enabled
            if name == "spectral_loss":
                if t.spectral_loss.enabled:
                    train_dict["spectral_loss"] = {"enabled": t.spectral_loss.enabled, "mode_size": t.spectral_loss.mode_size}
                continue
            if name == "imaml":
                if t.imaml.enabled:
                    imaml_dict = {
                        "enabled": True,
                        "lam": t.imaml.lam,
                        "lam_lr": t.imaml.lam_lr,
                        "lam_min": t.imaml.lam_min,
                        "cg_steps": t.imaml.cg_steps,
                        "cg_damping": t.imaml.cg_damping,
                        "inner_optimizer": t.imaml.inner_optimizer,
                        "outer_optimizer": t.imaml.outer_optimizer,
                        "outer_lbfgs_after": t.imaml.outer_lbfgs_after,
                        "anil": t.imaml.anil,
                        "anil_mode": t.imaml.anil_mode,
                        "proximal_every_step": t.imaml.proximal_every_step,
                    }
                    if t.imaml.slope_recovery_inner > 0:
                        imaml_dict["slope_recovery_inner"] = t.imaml.slope_recovery_inner
                    if t.imaml.slope_recovery_outer > 0:
                        imaml_dict["slope_recovery_outer"] = t.imaml.slope_recovery_outer
                    train_dict["imaml"] = imaml_dict
                continue

            # layers XOR old-format: only emit the format that was provided
            if name == "layers":
                if val is not None:
                    train_dict["layers"] = val
                continue
            if name in _layers_format:
                if t.layers is not None:
                    continue  # skip old-format fields when layers is set
                if val is not None:
                    train_dict[name] = val
                continue

            # Conv only if non-default
            if name in _conv:
                if t.conv_filters > 0:
                    train_dict[name] = val
                continue

            # Adaptive scales only when enabled
            if name == "adaptive_scales":
                if t.adaptive_scales:
                    train_dict[name] = True
                continue
            if name == "adaptive_scale_n":
                if t.adaptive_scales:
                    train_dict[name] = val
                continue

            if val is not None:
                train_dict[name] = val

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

    def to_network_config(self) -> "NetworkConfig":
        """Build NetworkConfig from the network fields in the training section."""
        from .networks.pde_operator_network import NetworkConfig

        t = self.training
        d: dict[str, Any] = {}
        if t.layers is not None:
            # Training-level adaptive_scales is a shortcut that also applies
            # to layers format: inject the flag into every hidden-layer entry
            # that doesn't already set one explicitly.
            if t.adaptive_scales:
                layers_list = [dict(entry) for entry in t.layers]
                for i, entry in enumerate(layers_list):
                    if i == 0 or i == len(layers_list) - 1:
                        continue  # skip input / output entries
                    if "hidden" in entry and "adaptive_scale" not in entry:
                        entry["adaptive_scale"] = True
                        entry["adaptive_scale_n"] = t.adaptive_scale_n
                d["layers"] = layers_list
            else:
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
            if t.adaptive_scales:
                d["adaptive_scales"] = True
                d["adaptive_scale_n"] = t.adaptive_scale_n
        if t.conv_filters > 0:
            d["conv_filters"] = t.conv_filters
            d["conv_kernel_size"] = t.conv_kernel_size
        if t.input_bypass:
            d["input_bypass"] = True
        return NetworkConfig.from_dict(d)

    # ── Convenience properties ───────────────────────────────────────

    @property
    def exp_dir(self) -> Path:
        """Experiment output directory."""
        return Path(self.output.base_dir) / self.experiment.name
