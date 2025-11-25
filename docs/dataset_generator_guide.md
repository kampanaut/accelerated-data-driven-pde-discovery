# Navier-Stokes Dataset Generator Guide

## Overview

This guide explains how to use the config-based Navier-Stokes dataset generator to create training data for PDE discovery with meta-learning.

## Quick Start

```bash
# Generate datasets from a config file
uv run python scripts/generate_ns_data.py --config configs/gaussian_sweep.yaml

# Test all IC types at once
uv run python scripts/generate_ns_data.py --config configs/all_ic_types_test.yaml
```

## What Gets Generated

For each initial condition in your config, the generator produces:

1. **Training data (.npz file)** containing:
   - Velocity fields: `u`, `v`
   - Spatial derivatives: `u_x`, `u_y`, `u_xx`, `u_yy`, `v_x`, `v_y`, `v_xx`, `v_yy`
   - Temporal derivatives: `u_t`, `v_t`
   - Coordinates: `x`, `y`, `t`
   - Metadata: `ic_config`, `simulation_params`

2. **Visualization (.png file)** showing flow evolution at 4 timesteps

## Available Initial Condition Types

### 1. Single Gaussian Vortex
Single rotating vortex with Gaussian profile.

```yaml
- name: "vortex_center"
  type: "gaussian_vortex"
  center: [3.14, 3.14]  # Position (x, y)
  width: 0.5            # Vortex width σ
  strength: 1.0         # Circulation Γ
```

**Vary:** position, width, strength

### 2. Multiple Vortices
Multiple vortices that can interact, merge, or orbit.

```yaml
- name: "three_vortices"
  type: "multi_vortex"
  vortices:
    - center: [2.0, 2.0]
      width: 0.4
      strength: 1.0      # Clockwise
    - center: [4.0, 4.0]
      width: 0.4
      strength: -1.0     # Counter-clockwise
    - center: [3.0, 4.5]
      width: 0.3
      strength: 0.8
```

**Vary:** number, positions, relative strengths, widths

### 3. Taylor-Green Vortex
Classic exact solution, periodic and symmetric.

```yaml
- name: "taylor_green"
  type: "taylor_green"
  amplitude: 1.0  # Velocity scale
```

**Vary:** amplitude

### 4. Lamb-Oseen Vortex
Realistic vortex with finite core (models actual vortices better than Gaussian).

```yaml
- name: "lamb_oseen"
  type: "lamb_oseen"
  center: [3.14, 3.14]
  core_radius: 0.6      # Core radius a
  circulation: 2.0      # Total circulation Γ
```

**Vary:** position, core radius, circulation

### 5. Dipole (Vortex Pair)
Two counter-rotating vortices that self-propel.

```yaml
- name: "dipole"
  type: "dipole"
  center: [3.14, 3.14]   # Midpoint
  separation: 1.5         # Distance between vortices
  width: 0.4
  strength: 1.0
```

**Vary:** separation, strength, width

### 6. Shear Layer (Kelvin-Helmholtz)
Velocity shear that becomes unstable and rolls up into vortices.

```yaml
- name: "kh_instability"
  type: "shear_layer"
  y_center: 3.14                # Location of shear
  thickness: 0.3                 # Transition thickness
  velocity_jump: 2.0             # Velocity difference
  perturbation_amplitude: 0.1    # Trigger instability
```

**Vary:** shear rate (velocity_jump), thickness, perturbation

### 7. Von Kármán Street
Alternating wake pattern (classic instability).

```yaml
- name: "karman_street"
  type: "von_karman"
  n_vortices: 8      # Total (split between 2 rows)
  spacing: 1.2       # Horizontal spacing
  offset: 2.0        # Vertical separation
  width: 0.4
  strength: 1.0
```

**Vary:** vortex spacing, offset, number

### 8. Random Vortex Soup
Multiple vortices at random positions - turbulent-like.

```yaml
- name: "soup_10vortices"
  type: "random_soup"
  n_vortices: 10
  strength_range: [-1.0, 1.0]
  width_range: [0.2, 0.5]
  seed: 42  # For reproducibility
```

**Vary:** number, strength distribution, seed

### 9. Perturbed Uniform Flow
Uniform flow with random perturbations (study transition to turbulence).

```yaml
- name: "perturbed_flow"
  type: "perturbed_flow"
  u_mean: 0.5
  v_mean: 0.0
  perturbation_amplitude: 0.1
  perturbation_wavelength: 2.0
```

**Vary:** mean flow, perturbation amplitude/wavelength

## Configuration File Structure

### Complete Example

```yaml
# Simulation parameters (shared across all ICs)
simulation:
  nu: 0.01                          # Viscosity
  domain_size: [6.283185, 6.283185] # 2π × 2π
  resolution: [32, 64]               # ny × nx grid
  t_end: 10.0                        # End time
  dt: 0.01                           # Timestep
  save_interval: 0.2                 # Snapshot frequency

# Output directory name
output_dir: "my_dataset"

# Initial conditions
initial_conditions:
  - name: "ic_001"
    type: "gaussian_vortex"
    center: [3.14, 3.14]
    width: 0.5
    strength: 1.0

  - name: "ic_002"
    type: "taylor_green"
    amplitude: 1.0

  # ... add more ICs
```

### Simulation Parameters

- `nu`: Kinematic viscosity (controls how fast vortices decay)
  - Higher ν → faster decay, smoother flow
  - Lower ν → longer-lived structures, more turbulent

- `domain_size`: Physical size `[Lx, Ly]`
  - Default: `[2π, 2π]` for periodic domain

- `resolution`: Grid size `[ny, nx]`
  - Higher resolution → more spatial detail, slower simulation
  - Current: `[32, 64]` (agreed with supervisor)

- `t_end`: How long to simulate
  - Longer time → see more decay/evolution

- `dt`: Integration timestep
  - Smaller dt → more accurate, slower
  - Rule of thumb: `dt < dx²/(4ν)` for stability

- `save_interval`: How often to save snapshots
  - Larger interval → fewer snapshots, smaller files

## Example Workflows

### Generate 3 Gaussian Vortices at Different Positions

```yaml
# configs/position_sweep.yaml
simulation:
  nu: 0.01
  domain_size: [6.283185, 6.283185]
  resolution: [32, 64]
  t_end: 10.0
  dt: 0.01
  save_interval: 0.2

output_dir: "position_sweep"

initial_conditions:
  - name: "center"
    type: "gaussian_vortex"
    center: [3.14, 3.14]
    width: 0.5
    strength: 1.0

  - name: "left"
    type: "gaussian_vortex"
    center: [1.57, 3.14]
    width: 0.5
    strength: 1.0

  - name: "bottom"
    type: "gaussian_vortex"
    center: [3.14, 1.57]
    width: 0.5
    strength: 1.0
```

Run: `uv run python scripts/generate_ns_data.py --config configs/position_sweep.yaml`

### Generate Strength Sweep

```yaml
# configs/strength_sweep.yaml
simulation:
  nu: 0.01
  domain_size: [6.283185, 6.283185]
  resolution: [32, 64]
  t_end: 10.0
  dt: 0.01
  save_interval: 0.2

output_dir: "strength_sweep"

initial_conditions:
  - name: "weak"
    type: "gaussian_vortex"
    center: [3.14, 3.14]
    width: 0.5
    strength: 0.5

  - name: "medium"
    type: "gaussian_vortex"
    center: [3.14, 3.14]
    width: 0.5
    strength: 1.0

  - name: "strong"
    type: "gaussian_vortex"
    center: [3.14, 3.14]
    width: 0.5
    strength: 2.0
```

### Generate Diverse Dataset for Meta-Learning

Mix different IC types to train a model that generalizes across flow patterns:

```yaml
# configs/diverse_meta.yaml
simulation:
  nu: 0.01
  domain_size: [6.283185, 6.283185]
  resolution: [32, 64]
  t_end: 10.0
  dt: 0.01
  save_interval: 0.2

output_dir: "diverse_meta"

initial_conditions:
  # Vortices
  - name: "single_vortex"
    type: "gaussian_vortex"
    center: [3.14, 3.14]
    width: 0.5
    strength: 1.0

  - name: "dipole"
    type: "dipole"
    center: [3.14, 3.14]
    separation: 1.5
    width: 0.4
    strength: 1.0

  - name: "multi_vortex"
    type: "multi_vortex"
    vortices:
      - {center: [2.0, 2.0], width: 0.4, strength: 1.0}
      - {center: [4.0, 4.0], width: 0.4, strength: -1.0}

  # Structured patterns
  - name: "taylor_green"
    type: "taylor_green"
    amplitude: 1.0

  - name: "von_karman"
    type: "von_karman"
    n_vortices: 8
    spacing: 1.2
    offset: 2.0
    width: 0.4
    strength: 1.0

  # Instabilities
  - name: "shear_layer"
    type: "shear_layer"
    y_center: 3.14
    thickness: 0.3
    velocity_jump: 2.0
    perturbation_amplitude: 0.1

  # Turbulent-like
  - name: "random_soup"
    type: "random_soup"
    n_vortices: 10
    strength_range: [-1.0, 1.0]
    width_range: [0.3, 0.5]
    seed: 42
```

Run: `uv run python scripts/generate_ns_data.py --config configs/diverse_meta.yaml`

This generates 7 datasets from different IC families - ideal for meta-learning.

## Output Structure

After running, you'll have:

```
data/
└── <output_dir>/
    ├── ic_001.npz                  # Training data
    ├── ic_001_evolution.png        # Visualization
    ├── ic_002.npz
    ├── ic_002_evolution.png
    └── ...
```

### Loading Data

```python
import numpy as np

# Load dataset
data = np.load('data/my_dataset/ic_001.npz')

# Access fields
u = data['u']          # shape: (n_samples,)
v = data['v']
u_x = data['u_x']      # ∂u/∂x
u_y = data['u_y']      # ∂u/∂y
u_xx = data['u_xx']    # ∂²u/∂x²
u_yy = data['u_yy']    # ∂²u/∂y²
u_t = data['u_t']      # ∂u/∂t (target)

# Coordinates
x_coord = data['x_coord']
y_coord = data['y_coord']
t = data['t']

# Metadata
ic_config = data['ic_config'].item()
sim_params = data['simulation_params'].item()
x = data['x']  # Grid x-coordinates
y = data['y']  # Grid y-coordinates

print(f"Dataset has {len(u):,} samples")
print(f"IC type: {ic_config['type']}")
```

## Pre-made Config Files

Located in `configs/`:

1. `gaussian_sweep.yaml` - 3 Gaussian vortices at different positions
2. `taylor_green.yaml` - Taylor-Green vortex test case
3. `shear_layer.yaml` - Kelvin-Helmholtz instability
4. `dipole.yaml` - Vortex pair
5. `von_karman.yaml` - Vortex street wake pattern
6. `random_soup.yaml` - Turbulent-like (5, 10, 15 vortices)
7. `mixed_ic_types.yaml` - Mix of 5 different IC types
8. `all_ic_types_test.yaml` - All 9 IC types in one run

## Tips for Creating Training Data

### For Meta-Learning (MAML)

Generate **diverse** datasets that vary in:
- IC type (different flow patterns)
- Parameters (position, strength, etc.)
- Complexity (single vortex → multi-vortex → turbulent)

**Recommended:** 20-50 different ICs mixing all types.

### For Single-Task Learning

If training on one PDE system:
- Generate many samples from similar ICs
- Vary parameters smoothly (position grid, strength sweep)

### Computational Considerations

**Memory usage per dataset:**
- `n_samples = (n_timesteps - 2) × (nx × ny)`
- For `[32, 64]` grid with 51 snapshots: ~100k samples
- Each sample has 12 features → ~10 MB per dataset

**Time per dataset:**
- Depends on `t_end / dt` and resolution
- Current settings: ~10-20 seconds per IC

## Troubleshooting

### "Config file not found"
- Check path is correct
- Use absolute path or run from project root

### Simulation diverges (NaN values)
- Reduce `dt` (timestep too large)
- Increase `nu` (viscosity too low for resolution)
- Check IC isn't too extreme (very high strength)

### Too many/too few snapshots
- Adjust `save_interval`
- Check `t_end` is appropriate for the flow

### Disk space issues
- Reduce number of ICs per config
- Increase `save_interval` (fewer timesteps)
- Lower resolution (but degrades quality)

## Next Steps

After generating data:

1. **Inspect visualizations** - Check that flows look physically reasonable
2. **Load and explore** - Look at derivative magnitudes, spatial patterns
3. **Train N network** - Use data to train PDE discovery model
4. **Implement MAML** - Use multiple datasets for meta-learning

## Technical Details

### Derivative Computation

**Spatial derivatives:** Central finite differences
- First: `u_x[i,j] ≈ (u[i,j+1] - u[i,j-1]) / (2Δx)`
- Second: `u_xx[i,j] ≈ (u[i,j+1] - 2u[i,j] + u[i,j-1]) / (Δx)²`

**Temporal derivatives:** Central finite differences across timesteps
- `u_t[t] ≈ (u[t+Δt] - u[t-Δt]) / (2Δt)`
- Loses first and last timesteps

### PDE Being Solved

2D incompressible Navier-Stokes:
```
∂u/∂t + u·∇u = -∇p + ν∇²u
∇·u = 0
```

Solved using PhiFlow with:
- Semi-Lagrangian advection
- Explicit diffusion
- Pressure projection for incompressibility
- Periodic boundary conditions

### Coordinate Convention

- `x`: horizontal (axis 1)
- `y`: vertical (axis 0)
- Arrays are `(ny, nx)` shape
- Domain: `[0, Lx] × [0, Ly]`

## References

- Deep Hidden Physics Models paper: `docs/3. Deep Hidden Physics Models.pdf`
- Finite differences explanation: `docs/finite_differences_and_ics.md`
- Concepts explained simply: `docs/concepts_explained_simply.md`
