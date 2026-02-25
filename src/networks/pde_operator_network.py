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
from typing import List


class PDEOperatorNetwork(nn.Module):
    """
    General-purpose neural network for learning PDE operators.

    Architecture is PDE-agnostic - it learns any mapping from
    (state + derivatives) → (time derivatives).


    Examples:
        # Navier-Stokes (default): 10 → 2
        net = PDEOperatorNetwork()

        # Burgers equation: 3 → 1
        net = PDEOperatorNetwork(input_dim=3, output_dim=1)

        # Custom architecture with pyramid structure
        net = PDEOperatorNetwork(hidden_dims=[128, 64, 32])

        # Bottleneck architecture
        net = PDEOperatorNetwork(hidden_dims=[64, 128, 64])
    """

    combo_indices: torch.Tensor

    def __init__(
        self,
        hidden_dims: List[int],
        input_dim: int = 10,
        output_dim: int = 2,
        activation: str = "tanh",
        conv_filters: int = 0,
        conv_kernel_size: int = 3,
    ):
        """
        Initialize PDE Operator Network.

        Args:
            input_dim: Number of input features
                - Navier-Stokes (2D): 10 (u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy)
            hidden_dims: List of hidden layer widths (e.g., [100, 100])
            output_dim: Number of outputs
            activation: Activation function ('tanh', 'relu', 'silu', 'gelu', 'mish')
            conv_filters: Number of conv filters for combinatorial feature expansion.
                0 = no conv (plain MLP). Features are reshaped to (n_fields, n_derivs)
                and all C(n_derivs, kernel_size) combinations are convolved.
            conv_kernel_size: Size of the combinatorial conv kernel.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.hidden_dims = hidden_dims if hidden_dims is not None else [100, 100]
        self.conv_filters = conv_filters

        # Choose activation function
        activation_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "silu": nn.SiLU,  # Swish: x * sigmoid(x), C^inf, unbounded
            "gelu": nn.GELU,  # Gaussian error linear unit, C^inf
            "mish": nn.Mish,  # x * tanh(softplus(x)), C^inf
        }
        if activation not in activation_map:
            raise ValueError(
                f"Unknown activation: {activation}. Use one of: {list(activation_map.keys())}"
            )
        act_fn = activation_map[activation]

        # Combinatorial conv setup
        if conv_filters > 0:
            n_fields = 2  # u, v
            n_derivs = input_dim // n_fields  # 5: [field, d_x, d_y, d_xx, d_yy]
            # All C(n_derivs, kernel_size) index triples (with_replacement=False)
            triples = torch.combinations(torch.arange(n_derivs), r=conv_kernel_size)
            self.register_buffer("combo_indices", triples)  # (n_combos, kernel_size)
            n_combos = len(triples)
            self.conv = nn.Conv1d(
                in_channels=n_fields,
                out_channels=conv_filters,
                kernel_size=conv_kernel_size,
                stride=conv_kernel_size,
                bias=True,
            )
            effective_input_dim = conv_filters * n_combos
        else:
            effective_input_dim = input_dim

        # Build MLP layers
        layers = []

        # Input layer: effective_input_dim → hidden_dims[0]
        layers.append(nn.Linear(effective_input_dim, self.hidden_dims[0]))
        layers.append(act_fn())

        # Hidden layers: hidden_dims[i] → hidden_dims[i+1]
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(act_fn())

        # Output layer: hidden_dims[-1] → output_dim (no activation)
        layers.append(nn.Linear(self.hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict time derivatives from state and spatial derivatives.

        Args:
            x: Input tensor of shape (batch_size, 10) containing:
               [u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]

        Returns:
            Output tensor of shape (batch_size, 2) containing:
            [u_t, v_t] (predicted time derivatives)
        """
        if self.conv_filters > 0:
            x_2d = torch.stack([
                x[:,[0, 2, 3, 4, 5]],
                x[:,[1, 6, 7, 8, 9]]
            ], dim=1) # (batch, 10) to (batch, 2, 5)
            x_2d = x_2d[:, :, self.combo_indices] # turn (batch, 2, 5) to (batch, 2, n_combos, kernel_size)

            x_2d = x_2d.flatten(2) # (batch, 2, n_combos * kernel_size)
            filtered = self.conv(x_2d) # (batch, filters, n_combos)
            x = filtered.flatten(1) # (batch, filters * n_combos)

        return self.network(x)

    def __repr__(self) -> str:
        """String representation of the network."""
        lines = [
            f"PDEOperatorNetwork(",
            f"  input_dim={self.input_dim},",
            f"  hidden_dims={self.hidden_dims},",
            f"  output_dim={self.output_dim},",
            f"  activation='{self.activation_name}',",
        ]
        if self.conv_filters > 0:
            n_combos = len(self.combo_indices)
            effective = self.conv_filters * n_combos
            lines.append(f"  conv=Conv1d({self.conv.in_channels}→{self.conv_filters}, C({self.input_dim // 2},{self.conv.kernel_size[0]})={n_combos} combos),")
            lines.append(f"  architecture={self.input_dim} → [conv {effective}] → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim},")
        else:
            lines.append(f"  architecture={self.input_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim},")
        lines.append(f"  total_params={sum(p.numel() for p in self.parameters()):,}")
        lines.append(f")")
        return "\n".join(lines)
