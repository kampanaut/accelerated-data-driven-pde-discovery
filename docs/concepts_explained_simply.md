# Core Concepts Explained Simply

A guide to understanding vorticity, Poisson equations, central differences, and how they appear in the code.

---

## 1. What is Vorticity?

### The Simple Picture

**Vorticity = how fast the fluid is spinning at a point.**

Imagine you drop a tiny paddlewheel into water. If the water is rotating, the wheel spins. **The spin rate = vorticity.**

```
     ↻ω > 0         ω = 0         ↺ω < 0
  (spinning CCW)  (no spin)   (spinning CW)

      ╔════╗       ─────→      ╔════╗
      ║ ↻  ║       ─────→      ║ ↺  ║
      ╚════╝       ─────→      ╚════╝
```

- **ω > 0**: Counter-clockwise rotation (like unscrewing with your right hand)
- **ω = 0**: No local spinning (could be flowing, but not rotating)
- **ω < 0**: Clockwise rotation (like screwing in)

### The Math

$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

**Breaking this down:**

- $u$ = velocity in x-direction (horizontal)
- $v$ = velocity in y-direction (vertical)
- $\frac{\partial v}{\partial x}$ = "how much does vertical velocity change as you move horizontally"
- $\frac{\partial u}{\partial y}$ = "how much does horizontal velocity change as you move vertically"

**Why this measures rotation:**

Think of a small box of fluid:

```
Top edge moving right: if u increases upward → box tilts
Right edge moving up:  if v increases rightward → box tilts

Net tilt = rotation!
```

If the right side moves up faster than the left side → rotation.
If the top moves right faster than the bottom → rotation (opposite direction).

**Vorticity = difference between these two tilting effects.**

### In Your Code

**File:** `src/data/initial_conditions.py:47-48`

```python
r_squared = (X - center[0])**2 + (Y - center[1])**2
vorticity = strength * np.exp(-r_squared / (2 * width**2))
```

**What this does:**

- `r_squared`: Distance squared from the vortex center
- `exp(-r²/width²)`: Gaussian bell curve - big at center, small far away
- `strength * ...`: Multiply by strength parameter

**Result:** A bump of rotation centered at `center`, with most rotation concentrated within distance `width`.

**Example:**
- `center = (π, π)` → rotation centered at domain middle
- `strength = 1.0` → rotation rate of 1.0 at the center
- `width = 0.5` → most rotation happens within radius ~0.5

---

## 2. Why Did We Choose Central Differences?

### What Are Finite Differences?

**Derivative** = rate of change = slope.

If you have discrete data points, you approximate the slope using nearby points.

### Three Ways to Approximate

Suppose you have data: `[..., u₂, u₃, u₄, ...]` at positions spaced by `Δx`.

#### Option 1: Forward Difference
$$\frac{du}{dx} \approx \frac{u_4 - u_3}{\Delta x}$$

Look ahead to the next point.

```
     u₃       u₄
     •--------•
         slope = (u₄-u₃)/Δx
```

**Accuracy:** Error ∼ Δx (first-order)

#### Option 2: Backward Difference
$$\frac{du}{dx} \approx \frac{u_3 - u_2}{\Delta x}$$

Look back to the previous point.

```
     u₂       u₃
     •--------•
         slope = (u₃-u₂)/Δx
```

**Accuracy:** Error ∼ Δx (first-order)

#### Option 3: Central Difference (What We Use)
$$\frac{du}{dx} \approx \frac{u_4 - u_2}{2\Delta x}$$

Look at both sides.

```
     u₂       u₃       u₄
     •--------•--------•
            slope = (u₄-u₂)/(2Δx)
```

**Accuracy:** Error ∼ (Δx)² (second-order)

### Why Central is Better

**Practical example:**

Grid spacing `Δx = 0.1`:

- **Forward/Backward error:** ~0.1 (10% of grid spacing)
- **Central error:** ~0.01 (1% of grid spacing)

**Central is 10× more accurate** for the same amount of data!

### Why Does Central Win?

When you do the Taylor series math (expanding u₄ and u₂ around u₃), the error terms that are proportional to Δx **cancel out**. Only the (Δx)² terms remain.

**Think of it like this:**
- Forward: biased toward one side
- Backward: biased toward other side
- Central: balanced → biases cancel

### When NOT to Use Central

1. **At boundaries:** Can't look both ways if you're at the edge of the domain
2. **For time-stepping:** Forward is "causal" (doesn't need future data)
3. **At discontinuities:** Central smooths across jumps (can be bad)

**For PDE data generation?** Central everywhere (except boundaries).

---

## 3. Where Do We See These Concepts in the Code?

Let me trace what happens when you run the script.

### Timeline of Events

```
1. You specify IC parameters → creates VORTICITY
2. Solve Poisson equation → get STREAM FUNCTION
3. Take derivatives (CENTRAL) → get VELOCITY
4. PhiFlow evolves velocity → many SNAPSHOTS
5. Take derivatives (CENTRAL) → get SPATIAL DERIVATIVES
6. Take derivatives (CENTRAL) → get TEMPORAL DERIVATIVES
7. Compute vorticity (for viz) → back to VORTICITY
```

Let me show you exactly where each happens.

---

### Event 1: Creating Vorticity

**File:** `scripts/generate_ns_data.py` lines 142-149

You write:
```python
ic_params = {
    'center': (np.pi, np.pi),  # Where the vortex is
    'width': 0.5,               # How wide the spinning region is
    'strength': 1.0,            # How fast it spins
}
```

**File:** `src/data/initial_conditions.py` lines 47-48

Code executes:
```python
# Compute distance from center
r_squared = (X - center[0])**2 + (Y - center[1])**2

# Create Gaussian bump of rotation
vorticity = strength * np.exp(-r_squared / (2 * width**2))
```

**What this creates:**

Imagine a top-down view of the domain:

```
     Low ω          High ω         Low ω
     ↓              ↓              ↓
   [0.01] ───→ [0.37] ───→ [1.0] ───→ [0.37] ───→ [0.01]
                        (center)
```

At center: ω = 1.0 (maximum spinning)
At edge: ω ≈ 0 (no spinning)

**This is "VORTICITY" - you've specified where the fluid should spin.**

---

### Event 2: Solving Poisson Equation

**File:** `src/data/initial_conditions.py` lines 50-51

```python
# Solve Poisson equation to get stream function
psi = solve_poisson_2d(vorticity, x, y)
```

This jumps to `solve_poisson_periodic()` at lines 101-135.

**What the Poisson equation does:**

You have: **Rotation field (ω)**
You want: **Velocity field (u, v)**

But you can't just randomly assign velocities - they must satisfy physics (incompressibility: no compression/expansion).

**Solution:** Solve for an intermediate quantity called **stream function (ψ)**.

The equation:
$$\nabla^2 \psi = \omega$$

Symbols:
- $\psi$ = stream function (what we're solving for)
- $\omega$ = vorticity (what you specified)
- $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ = "Laplacian" = "sum of second derivatives"

**In code (lines 115-135):**

```python
# Convert to Fourier space (frequencies)
kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
ky = 2 * np.pi * np.fft.fftfreq(ny, dy)

# In Fourier space, ∇² becomes multiplication by -k²
k_squared = KX**2 + KY**2

# Transform vorticity to Fourier space
rhs_hat = np.fft.fft2(rhs)

# Solve: ψ̂ = -ω̂ / k²
phi_hat = -rhs_hat / k_squared

# Transform back to real space
phi = np.fft.ifft2(phi_hat).real
```

**What's happening:**

Think of Fourier transform like decomposing a song into individual notes.
- Time domain: you hear the full song
- Frequency domain: you see which notes are playing

In frequency domain, derivatives become simple algebra. Solve there, then convert back.

**Result:** You now have `psi`, a 64×64 array of numbers. These are stream function values at each grid point.

---

### Event 3: Extract Velocity (Using Central Differences)

**File:** `src/data/initial_conditions.py` lines 57-58

```python
u = -np.gradient(psi, dy, axis=0)  # u = -∂ψ/∂y
v = np.gradient(psi, dx, axis=1)   # v = ∂ψ/∂x
```

**What `np.gradient` does:**

It implements central differences. For each interior point (i, j):

```python
# For u (derivative in y-direction)
u[i, j] = -(psi[i+1, j] - psi[i-1, j]) / (2 * dy)

# For v (derivative in x-direction)
v[i, j] = (psi[i, j+1] - psi[i, j-1]) / (2 * dx)
```

**Visual:**

```
Grid of psi values:

    j-1      j      j+1

i-1  •       •       •

i    • ←─── psi[i,j] ───→ •

i+1  •       •       •

To get v[i,j]: use psi[i, j+1] and psi[i, j-1]  (left-right)
To get u[i,j]: use psi[i+1, j] and psi[i-1, j]  (up-down)
```

**Result:** You now have velocity field `(u, v)` at each grid point that:
- Swirls around the vortex
- Is automatically incompressible (by construction)

**This is "CENTRAL DIFFERENCES" #1** - computing velocity from stream function.

---

### Event 4: PhiFlow Evolution

**File:** `src/pde/navier_stokes.py` lines 93-117

PhiFlow takes your initial velocity and evolves it forward in time:

```python
for step in range(1, 201):  # 200 timesteps
    # Advection: fluid moves along itself
    velocity = advect.semi_lagrangian(velocity, velocity, dt)

    # Diffusion: viscosity smooths things out
    velocity = diffuse.explicit(velocity, nu, dt)

    # Pressure projection: enforce incompressibility
    velocity, pressure = fluid.make_incompressible(velocity, ...)
```

**Result:** A list of 21 velocity snapshots at times [0, 0.1, 0.2, ..., 2.0].

---

### Event 5: Compute Spatial Derivatives (Using Central Differences)

**File:** `src/data/derivatives.py` lines 33-38

For each snapshot:

```python
def spatial_derivatives(field, dx, dy):
    # First derivatives
    field_x = np.gradient(field, dx, axis=1)  # ∂field/∂x
    field_y = np.gradient(field, dy, axis=0)  # ∂field/∂y

    # Second derivatives
    field_xx = np.gradient(field_x, dx, axis=1)  # ∂²field/∂x²
    field_yy = np.gradient(field_y, dy, axis=0)  # ∂²field/∂y²
```

**What's happening:**

For `u_x` at grid point (i, j):
```python
u_x[i, j] = (u[i, j+1] - u[i, j-1]) / (2 * dx)
```

For `u_xx` (second derivative):
```python
# First compute u_x everywhere
u_x[i, j] = (u[i, j+1] - u[i, j-1]) / (2 * dx)

# Then take derivative of u_x
u_xx[i, j] = (u_x[i, j+1] - u_x[i, j-1]) / (2 * dx)
```

Equivalently:
```python
u_xx[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (dx²)
```

**Visual for second derivative:**

```
     u[i, j-1]    u[i, j]    u[i, j+1]
         •──────────•──────────•

     u_xx measures "curvature" = how much does the slope change
```

**This is "CENTRAL DIFFERENCES" #2** - computing spatial derivatives from velocity snapshots.

---

### Event 6: Compute Temporal Derivatives (Using Central Differences)

**File:** `src/data/derivatives.py` lines 61-62

```python
for i in range(1, n_timesteps - 1):
    field_t[i - 1] = (field_history[i + 1] - field_history[i - 1]) / (2 * dt)
```

**What's happening:**

You have velocity at many times: `u(t=0), u(t=0.1), u(t=0.2), ..., u(t=2.0)`

To get time derivative at t=1.0:
```python
u_t(t=1.0) = (u(t=1.1) - u(t=0.9)) / (2 * 0.1)
```

**Visual:**

```
Timeline:

t=0.9     t=1.0     t=1.1
  •─────────•─────────•
  u₋₁       u₀        u₊₁

  slope = (u₊₁ - u₋₁) / (2Δt)
```

**Why we lose first and last timesteps:**

At t=0: Can't look backward (no t=-0.1)
At t=2.0: Can't look forward (no t=2.1)

So we only compute u_t for t ∈ [0.1, 1.9] → 19 timesteps with valid time derivatives.

**This is "CENTRAL DIFFERENCES" #3** - computing time derivatives across snapshots.

---

### Event 7: Compute Vorticity (For Visualization)

**File:** `src/data/derivatives.py` lines 97-98

```python
def compute_vorticity(u, v, dx, dy):
    v_x = np.gradient(v, dx, axis=1)  # ∂v/∂x
    u_y = np.gradient(u, dy, axis=0)  # ∂u/∂y
    return v_x - u_y                   # ω = ∂v/∂x - ∂u/∂y
```

**What's happening:**

We're going **backwards** from velocity → vorticity (opposite of ICs).

At each grid point:
```python
vorticity[i, j] = v_x[i, j] - u_y[i, j]
```

Using central differences:
```python
v_x[i, j] = (v[i, j+1] - v[i, j-1]) / (2 * dx)
u_y[i, j] = (u[i+1, j] - u[i-1, j]) / (2 * dy)
```

**This is used in:** `src/utils/visualization.py` line 77 to plot the vorticity contours you saw.

**This is "CENTRAL DIFFERENCES" #4 + "VORTICITY" #2** - computing rotation from evolved velocity field.

---

## 4. What is the Poisson Equation?

### The Formula

$$\nabla^2 \phi = f$$

**Breaking down the symbols:**

- $\phi$ = unknown function you're solving for (stream function in our case)
- $f$ = known "source" function (vorticity in our case)
- $\nabla^2$ = "del squared" = Laplacian operator = sum of second derivatives

**Expanded form:**
$$\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} = f$$

### What Does It Mean?

**Physical interpretation:** "The curvature of φ at a point equals the source density at that point."

Think of φ as a height map:
- Where f > 0: φ curves upward (like a hill)
- Where f < 0: φ curves downward (like a valley)
- Where f = 0: φ is flat

### Why Do We Solve It?

**The problem:** You want incompressible velocity.

**The constraint:**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

If you pick random u and v, they won't satisfy this.

**The solution:** Use stream function as an intermediary.

**The magic:** If you define velocity as:
$$u = -\frac{\partial \psi}{\partial y}, \quad v = \frac{\partial \psi}{\partial x}$$

Then incompressibility is **automatically satisfied**:
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = -\frac{\partial^2 \psi}{\partial x \partial y} + \frac{\partial^2 \psi}{\partial y \partial x} = 0$$

(The mixed derivatives cancel because ∂²ψ/∂x∂y = ∂²ψ/∂y∂x)

**But how to get ψ from desired rotation (ω)?**

That's the Poisson equation:
$$\nabla^2 \psi = \omega$$

### In Our Code

**File:** `src/data/initial_conditions.py` lines 101-135

**Input:** Vorticity field ω (64×64 array of rotation values)

**Output:** Stream function ψ (64×64 array)

**Method:** Fourier transform

```python
# Transform ω to frequency domain
omega_hat = fft2(omega)

# In frequency domain: ∇² becomes -k²
# So: -k² ψ̂ = ω̂
# Therefore: ψ̂ = -ω̂ / k²
psi_hat = -omega_hat / k_squared

# Transform back to real space
psi = ifft2(psi_hat)
```

**Why Fourier transform?**

Derivatives are hard in real space. They're easy in frequency space:
- Real space: ∂²ψ/∂x² requires finite differences (approximate)
- Frequency space: multiply by -k² (exact)

### Example Walkthrough

You specify:
```python
ω = 1.0 at center, → 0 far away (Gaussian bump)
```

Solve Poisson equation:
```python
∇²ψ = ω
```

Result: ψ looks like an upside-down Gaussian (negative bump).

Take derivatives:
```python
u = -∂ψ/∂y  →  creates swirling pattern
v = ∂ψ/∂x   →  creates swirling pattern
```

**Visualization:**

```
Vorticity ω:        Stream function ψ:        Velocity (u, v):

    High ω                Low ψ                   ↻ swirl
      ↓                     ↓                      ↓
   [1.0]    →  solve  →  [-5.0]  →  ∂/∂x,∂/∂y  → ╔════╗
   [0.5]                  [-2.5]                  ║ ↻  ║
   [0.0]                  [0.0]                   ╚════╝
```

---

## Summary: The Full Pipeline

### What You Specify
```
Vorticity: "I want rotation here"
```

### What Happens

1. **Solve Poisson** (∇²ψ = ω) → get stream function
2. **Take spatial derivatives** (central) → get velocity (u, v)
3. **Evolve with PhiFlow** → get velocity at many times
4. **Take spatial derivatives** (central) → get u_x, u_y, u_xx, u_yy
5. **Take temporal derivatives** (central) → get u_t, v_t
6. **Format as samples** → training data for N network

### Where Concepts Appear

| Concept | Where | What |
|---------|-------|------|
| **VORTICITY (specify)** | `initial_conditions.py:48` | `ω = strength * exp(...)` |
| **POISSON EQUATION** | `initial_conditions.py:128` | `ψ̂ = -ω̂ / k²` |
| **CENTRAL DIFF (velocity)** | `initial_conditions.py:57` | `u = -∂ψ/∂y` |
| **CENTRAL DIFF (spatial)** | `derivatives.py:33-38` | `u_x, u_xx, ...` |
| **CENTRAL DIFF (temporal)** | `derivatives.py:61` | `u_t = (u₊₁ - u₋₁)/(2Δt)` |
| **VORTICITY (compute)** | `derivatives.py:97` | `ω = v_x - u_y` |

### Key Insight

**You never directly specify velocity.** You specify rotation (vorticity), solve a mathematical problem (Poisson), and velocity emerges automatically - guaranteed to satisfy physics.

**Central differences appear everywhere** because they're the most accurate way to approximate derivatives from discrete data.

The whole pipeline is about converting your high-level specification ("I want a vortex here") into low-level training data (77,824 samples of derivatives).
