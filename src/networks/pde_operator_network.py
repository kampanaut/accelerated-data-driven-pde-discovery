"""
PDE Operator Network (N-network) - General purpose operator learning.

Learns the mapping: (state, spatial derivatives) → (time derivatives)

This network is PDE-agnostic and will be meta-trained with MAML to discover
PDE operators from data. Architecture follows Raissi et al. 2018.

Example PDEs:
- Navier-Stokes (2D): input_dim=10, output_dim=2
- Burgers (1D): input_dim=3, output_dim=1
- Brusselator: input_dim=6, output_dim=2
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any


# ── Layer spec dataclasses ──────────────────────────────────────────────────

@dataclass
class InputLayerSpec:
    """First entry in layers list. Declares input dimension."""
    input: int


@dataclass
class HiddenLayerSpec:
    """Middle entry in layers list. One hidden layer."""
    hidden: int
    activation: str | None = None
    bias: bool = True
    adaptive_scale: bool = False
    adaptive_scale_n: float = 1.0


@dataclass
class OutputLayerSpec:
    """Last entry in layers list. Output layer (no activation)."""
    output: int
    bias: bool = True


# ── Activation map ──────────────────────────────────────────────────────────


class Sin(nn.Module):
    """Wraps torch.sin as an nn.Module (Raissi 2018 DeepHPM activation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
    "sin": Sin,
}


class ScaledActivation(nn.Module):
    """N-LAAF wrapper from Jagtap, Kawaguchi, Karniadakis (Proc. R. Soc. A, 2020).

    Applies σ(n · a · preact) where:
      - preact is the affine preactivation (W·z + b) of a hidden layer
      - a is a per-neuron trainable scale of shape [num_features]
      - n is a fixed (non-trainable) scaling factor that controls the
        reachable slope range per unit movement of a

    Paper init rule: n · a = 1 at iteration 0, so the effective slope is
    identical to a vanilla activation. With fixed n, that means
    a_init = 1/n.
    """

    n: torch.Tensor  # registered buffer

    def __init__(self, base: nn.Module, num_features: int, n: float = 1.0):
        super().__init__()
        if n <= 0:
            raise ValueError(f"ScaledActivation requires n > 0, got {n}")
        self.base = base
        self.register_buffer("n", torch.tensor(float(n)))
        self.a = nn.Parameter(torch.full((num_features,), 1.0 / float(n)))

    def forward(self, preact: torch.Tensor) -> torch.Tensor:
        return self.base(self.n * self.a * preact)


# ── Network config ──────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """Normalized network architecture config.

    Constructed via NetworkConfig.from_dict(train_cfg), which accepts either:

    Old format (still supported):
        {"hidden_dims": [100, 100], "activation": "silu", "input_dim": 5, "output_dim": 1}

    New format (layers):
        {"layers": [{"input": 5}, {"hidden": 100, "activation": "silu"}, {"output": 1}]}

    The two formats are mutually exclusive (XOR).
    """
    input_spec: InputLayerSpec
    hidden_specs: list[HiddenLayerSpec] = field(default_factory=list)
    output_spec: OutputLayerSpec = field(default_factory=lambda: OutputLayerSpec(output=2))
    conv_filters: int = 0
    conv_kernel_size: int = 3
    input_bypass: bool = False

    @property
    def input_dim(self) -> int:
        return self.input_spec.input

    @property
    def output_dim(self) -> int:
        return self.output_spec.output

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkConfig":
        """Parse config dict into NetworkConfig. Validates XOR between formats."""
        has_layers = "layers" in d
        old_keys = {"hidden_dims", "input_dim", "output_dim", "activation"}
        has_old = any(k in d for k in old_keys)

        if has_layers and has_old:
            conflicting = old_keys & d.keys()
            raise ValueError(
                f"Cannot mix 'layers' with {conflicting}. Use one format or the other."
            )
        if not has_layers and not has_old:
            raise ValueError(
                "Must specify either 'layers' or 'hidden_dims'+'input_dim'+'output_dim'."
            )

        conv_filters = d.get("conv_filters", 0)
        conv_kernel_size = d.get("conv_kernel_size", 3)
        input_bypass = d.get("input_bypass", False)

        if has_layers:
            cfg = cls._parse_layers(d["layers"], conv_filters, conv_kernel_size)
        else:
            cfg = cls._parse_old(d, conv_filters, conv_kernel_size)
        cfg.input_bypass = input_bypass
        return cfg

    @classmethod
    def _parse_layers(
        cls, layers_list: list[dict[str, Any]], conv_filters: int, conv_kernel_size: int
    ) -> "NetworkConfig":
        if len(layers_list) < 2:
            raise ValueError("layers must have at least 2 entries (input + output)")

        # First entry: InputLayerSpec
        first = layers_list[0]
        if "input" not in first:
            raise ValueError(
                f"First layer must have 'input' key, got: {first}"
            )
        input_spec = InputLayerSpec(input=first["input"])

        # Last entry: OutputLayerSpec
        last = layers_list[-1]
        if "output" not in last:
            raise ValueError(
                f"Last layer must have 'output' key, got: {last}"
            )
        output_spec = OutputLayerSpec(
            output=last["output"],
            bias=last.get("bias", True),
        )

        # Middle entries: HiddenLayerSpec
        hidden_specs = []
        for i, layer in enumerate(layers_list[1:-1], start=1):
            if "hidden" not in layer:
                raise ValueError(
                    f"Layer {i} must have 'hidden' key (middle layers are hidden), got: {layer}"
                )
            act = layer.get("activation")
            if act is not None and act not in ACTIVATION_MAP:
                raise ValueError(
                    f"Unknown activation '{act}' in layer {i}. "
                    f"Use one of: {list(ACTIVATION_MAP.keys())}"
                )
            hidden_specs.append(HiddenLayerSpec(
                hidden=layer["hidden"],
                activation=act,
                bias=layer.get("bias", True),
                adaptive_scale=layer.get("adaptive_scale", False),
                adaptive_scale_n=layer.get("adaptive_scale_n", 1.0),
            ))

        return cls(input_spec, hidden_specs, output_spec, conv_filters, conv_kernel_size)

    @classmethod
    def _parse_old(cls, d: dict, conv_filters: int, conv_kernel_size: int) -> "NetworkConfig":
        input_dim = d.get("input_dim", 10)
        output_dim = d.get("output_dim", 2)
        hidden_dims = d.get("hidden_dims", [100, 100])
        activation = d.get("activation", "tanh")

        if activation not in ACTIVATION_MAP:
            raise ValueError(
                f"Unknown activation '{activation}'. Use one of: {list(ACTIVATION_MAP.keys())}"
            )

        adaptive_scales = d.get("adaptive_scales", False)
        adaptive_scale_n = d.get("adaptive_scale_n", 1.0)

        input_spec = InputLayerSpec(input=input_dim)
        output_spec = OutputLayerSpec(output=output_dim)
        hidden_specs = [
            HiddenLayerSpec(
                hidden=h,
                activation=activation,
                adaptive_scale=adaptive_scales,
                adaptive_scale_n=adaptive_scale_n,
            )
            for h in hidden_dims
        ]

        return cls(input_spec, hidden_specs, output_spec, conv_filters, conv_kernel_size)

    def to_dict(self) -> dict:
        """Serialize to dict in layers format (canonical form)."""
        layers: list[dict[str, Any]] = [{"input": self.input_spec.input}]
        for h in self.hidden_specs:
            entry: dict[str, Any] = {"hidden": h.hidden}
            if h.activation is not None:
                entry["activation"] = h.activation
            if not h.bias:
                entry["bias"] = False
            if h.adaptive_scale:
                entry["adaptive_scale"] = True
                entry["adaptive_scale_n"] = h.adaptive_scale_n
            layers.append(entry)

        out_entry: dict[str, Any] = {"output": self.output_spec.output}
        if not self.output_spec.bias:
            out_entry["bias"] = False
        layers.append(out_entry)

        d: dict[str, Any] = {"layers": layers}
        if self.conv_filters != 0:
            d["conv_filters"] = self.conv_filters
            d["conv_kernel_size"] = self.conv_kernel_size
        if self.input_bypass:
            d["input_bypass"] = True
        return d


# ── Network ─────────────────────────────────────────────────────────────────

class PDEOperatorNetwork(nn.Module):
    """
    General-purpose neural network for learning PDE operators.

    Architecture is PDE-agnostic - it learns any mapping from
    (state + derivatives) → (time derivatives).

    Examples:
        # Standard MLP (old format)
        config = NetworkConfig.from_dict({
            "hidden_dims": [100, 100], "activation": "silu",
            "input_dim": 5, "output_dim": 1,
        })
        net = PDEOperatorNetwork(config)

        # Same thing (new format)
        config = NetworkConfig.from_dict({"layers": [
            {"input": 5},
            {"hidden": 100, "activation": "silu"},
            {"hidden": 100, "activation": "silu"},
            {"output": 1},
        ]})
        net = PDEOperatorNetwork(config)

        # Linear model (cheat experiment)
        config = NetworkConfig.from_dict({"layers": [
            {"input": 5},
            {"output": 1, "bias": False},
        ]})
        net = PDEOperatorNetwork(config)
    """

    combo_indices: torch.Tensor

    def __init__(self, config: NetworkConfig):
        super().__init__()

        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.conv_filters = config.conv_filters

        # Combinatorial conv setup
        if config.conv_filters > 0:
            n_fields = 2  # u, v
            n_derivs = config.input_dim // n_fields
            triples = torch.combinations(torch.arange(n_derivs), r=config.conv_kernel_size)
            self.register_buffer("combo_indices", triples)
            n_combos = len(triples)
            self.conv = nn.Conv1d(
                in_channels=n_fields,
                out_channels=config.conv_filters,
                kernel_size=config.conv_kernel_size,
                stride=config.conv_kernel_size,
                bias=True,
            )
            prev_dim = config.conv_filters * n_combos
        else:
            prev_dim = config.input_dim

        # Build layers from config
        layers: list[nn.Module] = []

        for spec in config.hidden_specs:
            layers.append(nn.Linear(prev_dim, spec.hidden, bias=spec.bias))
            if spec.activation is not None:
                base_act = ACTIVATION_MAP[spec.activation]()
                if spec.adaptive_scale:
                    layers.append(ScaledActivation(base_act, spec.hidden, spec.adaptive_scale_n))
                else:
                    layers.append(base_act)
            prev_dim = spec.hidden

        # Input bypass: head receives body output + raw input
        self._input_bypass = config.input_bypass
        if config.input_bypass:
            head_input = prev_dim + config.input_dim
        else:
            head_input = prev_dim

        layers.append(nn.Linear(
            head_input, config.output_spec.output, bias=config.output_spec.bias
        ))

        if config.input_bypass:
            # Split into body (all but last) and head (last layer)
            self.body = nn.Sequential(*layers[:-1])
            self.head = layers[-1]
            self.network = None  # type: ignore[assignment]
        else:
            self.network = nn.Sequential(*layers)

        # Cache ScaledActivation references. These are already registered as
        # children via self.body / self.network, so storing them in a plain
        # list (not a ModuleList) avoids double-registration.
        self._scale_modules: list[ScaledActivation] = [
            m for m in self.modules() if isinstance(m, ScaledActivation)
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the output linear layer. Used for ANIL inner-loop selection."""
        if self._input_bypass:
            return list(self.head.parameters())
        assert self.network is not None
        return list(self.network[-1].parameters())

    def slope_recovery(self) -> torch.Tensor:
        """N-LAAF slope recovery term S(a) per Jagtap et al. 2020.

        Paper:  S(a) = 1 / [(1/L) · Σ_k exp(mean_i(a_i^k))]
        Here L counts ScaledActivation modules actually present in the network
        (may be fewer than D-1 if some hidden layers have adaptive_scale=False).

        Returns a scalar tensor on the network's device. Returns 0 if the
        network has no ScaledActivation modules.
        """
        if not self._scale_modules:
            return torch.zeros((), device=next(self.parameters()).device)
        layer_exps = torch.stack([torch.exp(m.a.mean()) for m in self._scale_modules])
        return layer_exps.numel() / layer_exps.sum()

    def adaptive_scale_parameters(self, mode: str = "all") -> list[nn.Parameter]:
        """Per-neuron N-LAAF scale parameters.

        Args:
            mode: "all"  → every ScaledActivation's `a` in the network.
                  "last" → only the last ScaledActivation's `a` (if any).
        """
        if mode not in ("all", "last"):
            raise ValueError(f"mode must be 'all' or 'last', got {mode!r}")
        if not self._scale_modules:
            return []
        if mode == "all":
            return [m.a for m in self._scale_modules]
        return [self._scale_modules[-1].a]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict time derivatives from state and spatial derivatives.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if self.conv_filters > 0:
            x_2d = torch.stack(
                [x[:, [0, 2, 3, 4, 5]], x[:, [1, 6, 7, 8, 9]]], dim=1
            )  # (batch, 10) to (batch, 2, 5)
            x_2d = x_2d[
                :, :, self.combo_indices
            ]  # turn (batch, 2, 5) to (batch, 2, n_combos, kernel_size)

            x_2d = x_2d.flatten(2)  # (batch, 2, n_combos * kernel_size)
            filtered = self.conv(x_2d)  # (batch, filters, n_combos)
            x = filtered.flatten(1)  # (batch, filters * n_combos)

        if self._input_bypass:
            body_out = self.body(x)
            return self.head(torch.cat([body_out, x], dim=-1))
        return self.network(x)  # type: ignore[misc]

    def __repr__(self) -> str:
        """String representation of the network."""
        cfg = self.config
        parts = []
        parts.append(str(cfg.input_dim))
        for h in cfg.hidden_specs:
            label = str(h.hidden)
            if h.activation:
                label += f", {h.activation}"
            if not h.bias:
                label += ", no bias"
            parts.append(f"[{label}]")
        out_label = str(cfg.output_dim)
        if not cfg.output_spec.bias:
            out_label += ", no bias"
        parts.append(out_label)

        lines = [
            "PDEOperatorNetwork(",
            f"  layers: {' → '.join(parts)},",
        ]
        if cfg.conv_filters > 0:
            n_combos = len(self.combo_indices)
            lines.append(
                f"  conv=Conv1d({self.conv.in_channels}→{cfg.conv_filters}, "
                f"C({cfg.input_dim // 2},{self.conv.kernel_size[0]})={n_combos} combos),"
            )
        lines.append(f"  total_params={sum(p.numel() for p in self.parameters()):,}")
        lines.append(")")
        return "\n".join(lines)
