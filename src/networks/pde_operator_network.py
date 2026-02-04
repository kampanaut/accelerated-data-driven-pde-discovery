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

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = None,
        output_dim: int = 2,
        activation: str = 'tanh'
    ):
        """
        Initialize PDE Operator Network.

        Args:
            input_dim: Number of input features
                - Navier-Stokes (2D): 10 (u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy)
            hidden_dims: List of hidden layer widths (e.g., [100, 100])
                - Default: [100, 100] (2 layers of 100 neurons each)
                - Pyramid: [128, 64, 32]
                - Bottleneck: [64, 128, 64]
            output_dim: Number of outputs
                - Navier-Stokes (2D): 2 (u_t, v_t)
                - Burgers (1D): 1 (u_t)
                - Custom: any integer
            activation: Activation function ('tanh' or 'relu')
                - 'tanh': Smooth, good for physics (default)
                - 'relu': Faster, more common in ML

        Example:
            # Default (2 layers × 100 neurons)
            net = PDEOperatorNetwork()

            # Custom widths (3 layers with different sizes)
            net = PDEOperatorNetwork(hidden_dims=[128, 64, 32])

            # Burgers equation (1D PDE)
            net = PDEOperatorNetwork(input_dim=3, hidden_dims=[50, 50], output_dim=1)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.hidden_dims = hidden_dims if hidden_dims is not None else [100, 100]

        # Choose activation function
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'tanh' or 'relu'.")

        # Build network layers
        layers = []

        # Input layer: input_dim → hidden_dims[0]
        layers.append(nn.Linear(input_dim, self.hidden_dims[0]))
        layers.append(act_fn())

        # Hidden layers: hidden_dims[i] → hidden_dims[i+1]
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
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
        return self.network(x)

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"PDEOperatorNetwork(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  output_dim={self.output_dim},\n"
            f"  activation='{self.activation_name}',\n"
            f"  architecture={self.input_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim},\n"
            f"  total_params={sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )
