# Meta-Learning for Accelerated PDE Discovery

Raissi [1] introduced the Deep Hidden Physics framework for data-driven PDE discovery, where a neural network — the PDE operator network — learns the nonlinear mapping from spatial derivatives to the time derivative:

$$u_t = \mathcal{N}(u, u_x, u_y, u_{xx}, u_{yy}, \ldots)$$

Once trained, the learned operator encodes the discovered physics, and its weights implicitly represent the PDE coefficients, which can be extracted via Jacobian analysis.

However, this approach requires training a new network from scratch for each system encountered. Every new set of physical parameters demands a full training run, failing to exploit structural similarities across related systems.

Model-Agnostic Meta-Learning (MAML) [2] offers a solution: rather than learning weights for one system, MAML learns a meta-parameter initialisation $\theta^*$ — a starting point from which the network can rapidly adapt to any system within a task family. Early results on the heat equation show that certain MAML configurations recover PDE coefficients closer to the true value than the baseline — not accurately, but with a clear relative improvement over random initialisation $\theta_0$. Other configurations exhibit two distinct failure modes: coefficients that collapse near zero (the network fits the data but encodes nothing extractable), or anti-tracking, where recovered coefficients systematically invert the sign of the true value — suggesting the network absorbs the coefficient into other terms with a compensating negative contribution. The non-meta-learned baseline consistently falls into one of these failure modes. These observations are specific to the heat equation (pure diffusion); the Brusselator (reaction-diffusion) has not yet shown the same separation.

## Quick Start

```bash
# Setup
uv sync

# 1. Generate data (Dedalus spectral solver)
uv run python scripts/generate_heat_data.py --config configs/heat_train-1.yaml
uv run python scripts/generate_heat_data.py --config configs/heat_test-1.yaml

# 2. Train
uv run python scripts/train_maml.py --config configs/formal/heat-17-1step-k800-baseline.yaml

# 3. Evaluate
uv run python scripts/evaluate.py --config configs/formal/heat-17-1step-k800-baseline.yaml

# 4. Visualize (cross-experiment scatter)
uv run python scripts/visualize.py --config configs/formal/heat-17-1step-k800-baseline.yaml

# 5. Post-eval filtering
uv run python scripts/post_eval_filter.py --pde heat
```

## Project Structure

```
src/
  pde/               # Dedalus v3 spectral solvers (IMEX RK222)
    brusselator.py, fitzhugh_nagumo.py, lambda_omega.py,
    heat_equation.py, nl_heat_equation.py, navier_stokes.py
  data/
    generation.py     # Dataset generation pipeline (PDESpec + FieldType)
    fourier_eval.py   # GPU-native Fourier feature/target evaluation
    initial_conditions_fhn.py, initial_conditions_lo.py
  training/
    task_loader.py    # PDETask ABC + CoefficientSpec + TASK_REGISTRY
    maml.py           # MAML training loop (requires `higher`)
  evaluation/
    jacobian.py       # JVP-based coefficient recovery
    metrics.py        # Plateau detection, WORSE flags, step compression
    graphs.py         # All plot functions
  networks/           # MLP architecture

scripts/
  generate_*_data.py  # Per-PDE data generation
  train_maml.py       # Training entry point
  evaluate.py         # Evaluation entry point (fine-tune + Jacobian)
  visualize.py        # Cross-experiment scatter plots
  post_eval_filter.py # Filter models by WORSE flag analysis
  dataset_analyser.py # SVD/correlation/collinearity analysis

configs/
  formal/             # Formal experiment grid (48 experiments)
  exploratory/        # Archived exploratory experiments
  *_train-*.yaml      # Data generation configs
  *_test-*.yaml
```

## Config & Seed Scheme

Each PDE gets a million-block seed: NS=1M, BR=2M, FHN=3M, LO=4M, Heat=5M, NLHeat=6M.

Formal experiment grid: `{pde}-{id}-{inner_steps}step-k{k_shot}-{loss_mode}.yaml`

Loss modes: `baseline`, `metal`, `spectral`, `metal-spectral`.

## Key Conventions

- Features: `[u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy]` (10 inputs)
- Targets: `[u_t, v_t]` (2 outputs, from PDE RHS — not temporal FD)
- Data format: Fourier only (`*_fourier.npz` with `u_hat/v_hat`, NS adds `p_hat`)
- All GPU-native torch pipeline, numpy only at `np.load` boundary
- `uv run` for everything
