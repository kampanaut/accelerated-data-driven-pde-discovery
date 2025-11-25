# Finite Differences and Initial Conditions Explained

## Part 1: How Finite Differences Work

### The Basic Idea

A derivative measures how fast something changes. Finite differences approximate this by looking at nearby points.

**Definition of derivative:**
$$\frac{du}{dx} = \lim_{h \to 0} \frac{u(x+h) - u(x)}{h}$$

**Finite difference approximation:**
Just use a small but finite $h = \Delta x$ instead of taking the limit to zero:
$$\frac{du}{dx} \approx \frac{u(x+\Delta x) - u(x)}{\Delta x}$$

---

### Three Types of Finite Differences

#### 1. Forward Difference
$$\frac{du}{dx} \approx \frac{u_{i+1} - u_i}{\Delta x}$$

Uses the point ahead. **Accuracy:** $O(\Delta x)$ (first-order accurate).

#### 2. Backward Difference
$$\frac{du}{dx} \approx \frac{u_i - u_{i-1}}{\Delta x}$$

Uses the point behind. **Accuracy:** $O(\Delta x)$ (first-order accurate).

#### 3. Central Difference (what we use)
$$\frac{du}{dx} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}$$

Uses points on both sides. **Accuracy:** $O(\Delta x^2)$ (second-order accurate).

**Why central is better:** If you expand $u_{i+1}$ and $u_{i-1}$ in Taylor series:
$$u_{i+1} = u_i + u'_i \Delta x + \frac{1}{2}u''_i (\Delta x)^2 + \frac{1}{6}u'''_i (\Delta x)^3 + \ldots$$
$$u_{i-1} = u_i - u'_i \Delta x + \frac{1}{2}u''_i (\Delta x)^2 - \frac{1}{6}u'''_i (\Delta x)^3 + \ldots$$

Subtract them:
$$u_{i+1} - u_{i-1} = 2u'_i \Delta x + \frac{1}{3}u'''_i (\Delta x)^3 + \ldots$$

Divide by $2\Delta x$:
$$\frac{u_{i+1} - u_{i-1}}{2\Delta x} = u'_i + O((\Delta x)^2)$$

The $O((\Delta x)^2)$ terms cancel! Much more accurate than forward/backward.

---

### Second Derivatives

For $\frac{d^2u}{dx^2}$, use the **three-point stencil:**
$$\frac{d^2u}{dx^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{(\Delta x)^2}$$

**Derivation:** Add the Taylor expansions instead of subtracting:
$$u_{i+1} + u_{i-1} = 2u_i + u''_i (\Delta x)^2 + O((\Delta x)^4)$$

Rearrange:
$$u''_i = \frac{u_{i+1} - 2u_i + u_{i-1}}{(\Delta x)^2} + O((\Delta x)^2)$$

Also second-order accurate.

---

### Example: Computing Derivatives from Data

You have velocity $u$ on a grid: $[u_0, u_1, u_2, u_3, u_4]$ with spacing $\Delta x = 0.1$.

**First derivative at $i=2$:**
$$u'_2 \approx \frac{u_3 - u_1}{2 \times 0.1} = \frac{u_3 - u_1}{0.2}$$

**Second derivative at $i=2$:**
$$u''_2 \approx \frac{u_3 - 2u_2 + u_1}{(0.1)^2} = \frac{u_3 - 2u_2 + u_1}{0.01}$$

**Temporal derivative:** Same idea, but across time:
$$\frac{du}{dt} \bigg|_{t=1.0} \approx \frac{u(t=1.1) - u(t=0.9)}{2 \times 0.1}$$

Need three consecutive snapshots: before, current, after.

---

### Why This Works for Navier-Stokes

Navier-Stokes equation:
$$\frac{\partial u}{\partial t} = -u\frac{\partial u}{\partial x} - v\frac{\partial u}{\partial y} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) - \frac{\partial p}{\partial x}$$

**Right side:** All spatial derivatives ($u_x, u_y, u_{xx}, u_{yy}, p_x$).
**Left side:** Time derivative ($u_t$).

If you have the velocity field $u(t, x, y)$ from a solver:
1. Compute spatial derivatives using finite differences on the grid
2. Compute temporal derivatives using finite differences across timesteps
3. Now you have all the ingredients to train the N network

The N network learns: "Given these spatial derivatives → predict the time derivative."

---

## Part 2: Initial Conditions

### What Are Initial Conditions?

Initial conditions (ICs) specify the **starting state** of the system at $t=0$.

For Navier-Stokes: $u(x, y, t=0) = ?$ and $v(x, y, t=0) = ?$

The PDE tells you how the system **evolves** from this starting state, but you need to specify where it starts.

---

### Why Vorticity for ICs?

**Problem:** You can't just set arbitrary $(u, v)$ because of incompressibility:
$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

If you pick random $u$ and $v$, they probably won't satisfy this constraint.

**Solution:** Use vorticity and stream function.

---

### The Vorticity-Stream Function Approach

**Step 1: Define vorticity**
$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

Vorticity measures local rotation of the fluid.

**Step 2: Relate stream function to velocity**
$$u = -\frac{\partial \psi}{\partial y}, \quad v = \frac{\partial \psi}{\partial x}$$

**Key property:** This automatically satisfies incompressibility!
$$\nabla \cdot \mathbf{u} = -\frac{\partial^2 \psi}{\partial x \partial y} + \frac{\partial^2 \psi}{\partial y \partial x} = 0$$

(Mixed partials are equal, so they cancel.)

**Step 3: Relate stream function to vorticity**

Take curl of velocity definition:
$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} = \frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = \nabla^2 \psi$$

So:
$$\boxed{\nabla^2 \psi = \omega}$$

This is a **Poisson equation**.

---

### Creating Initial Conditions: The Recipe

**What you control:** The vorticity field $\omega(x, y)$ at $t=0$.

**Steps:**
1. Choose a vorticity distribution (e.g., Gaussian bump)
2. Solve Poisson equation: $\nabla^2 \psi = \omega$ for stream function $\psi$
3. Compute velocity: $u = -\partial \psi / \partial y$, $v = \partial \psi / \partial x$
4. You now have $(u, v)$ that is guaranteed to be incompressible

---

### Example: Gaussian Vortex

**Vorticity specification:**
$$\omega(x, y) = A \exp\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2}\right)$$

Where:
- $(x_0, y_0)$ = vortex center
- $\sigma$ = width of the vortex
- $A$ = strength (circulation)

**Physical meaning:** A concentrated rotating blob at $(x_0, y_0)$.

**Solve Poisson equation:**
$$\frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial y^2} = \omega(x, y)$$

This is done numerically (scipy sparse solver in the code).

**Extract velocity:**
$$u = -\frac{\partial \psi}{\partial y} \approx -\frac{\psi_{i,j+1} - \psi_{i,j-1}}{2\Delta y}$$
$$v = \frac{\partial \psi}{\partial x} \approx \frac{\psi_{i+1,j} - \psi_{i-1,j}}{2\Delta x}$$

**Result:** Velocity field that swirls around $(x_0, y_0)$.

---

### Creating Your Own Initial Conditions

You can create any IC by specifying $\omega(x, y)$. Examples:

#### 1. Multiple Vortices
$$\omega(x, y) = A_1 e^{-r_1^2/\sigma_1^2} + A_2 e^{-r_2^2/\sigma_2^2}$$

where $r_1 = \sqrt{(x-x_1)^2 + (y-y_1)^2}$, etc.

Creates multiple swirling regions. If $A_1$ and $A_2$ have opposite signs, they rotate in opposite directions.

#### 2. Taylor-Green Vortex (exact solution)
$$u(x, y, t=0) = -\sin(x)\cos(y)$$
$$v(x, y, t=0) = \cos(x)\sin(y)$$

Classic test case. Already incompressible by construction.

#### 3. Shear Layer
$$\omega(x, y) = A \tanh\left(\frac{y - y_0}{\delta}\right)$$

Creates a velocity jump across $y = y_0$ (Kelvin-Helmholtz instability setup).

#### 4. Random Turbulence
$$\omega(x, y) = \sum_{k_x, k_y} \hat{\omega}_{k_x, k_y} e^{i(k_x x + k_y y)}$$

where $\hat{\omega}$ are random Fourier coefficients with specified energy spectrum.

#### 5. Point Vortices
$$\omega(x, y) = \Gamma \delta(x - x_0, y - y_0)$$

Idealized infinitesimal vortex (need regularization for numerics).

---

### Code Example: Custom IC

Want a **double vortex** (two counter-rotating vortices)?

```python
from src.data.initial_conditions import multi_vortex_ic
import numpy as np

# Define domain
x = np.linspace(0, 2*np.pi, 64)
y = np.linspace(0, 2*np.pi, 64)

# Define two vortices
vortex_params = [
    {'center': (np.pi/2, np.pi), 'width': 0.3, 'strength': 1.0},   # Clockwise
    {'center': (3*np.pi/2, np.pi), 'width': 0.3, 'strength': -1.0}  # Counter-clockwise
]

# Generate IC
u, v = multi_vortex_ic(vortex_params, x, y)
```

This creates two vortices rotating in opposite directions. They'll interact and create interesting dynamics.

---

### How to Think About ICs

**Vorticity is the "input knob":**
- Positive $\omega$ → counter-clockwise rotation
- Negative $\omega$ → clockwise rotation
- Magnitude → strength of rotation
- Spatial distribution → shape of the flow

**Velocity is the "output":**
- Automatically incompressible
- Determined by solving Poisson equation

**For meta-learning:**
- Generate many datasets with different $\omega$ distributions
- N network learns the PDE dynamics that work across all of them
- MAML optimizes for fast adaptation to new $\omega$ configurations

---

## Part 3: What Happens in Your Code

### In `initial_conditions.py`:

```python
def gaussian_vortex_ic(center, width, strength, x, y):
    # 1. Create vorticity field
    r_squared = (X - center[0])**2 + (Y - center[1])**2
    vorticity = strength * np.exp(-r_squared / (2 * width**2))

    # 2. Solve Poisson equation: ∇²ψ = -ω
    psi = solve_poisson_2d(vorticity, x, y)

    # 3. Extract velocity: u = -∂ψ/∂y, v = ∂ψ/∂x
    u = -np.gradient(psi, dy, axis=0)
    v = np.gradient(psi, dx, axis=1)

    return u, v
```

### In `derivatives.py`:

```python
def spatial_derivatives(field, dx, dy):
    # Central differences for first derivatives
    u_x = np.gradient(field, dx, axis=1)  # ∂u/∂x
    u_y = np.gradient(field, dy, axis=0)  # ∂u/∂y

    # Apply gradient again for second derivatives
    u_xx = np.gradient(u_x, dx, axis=1)   # ∂²u/∂x²
    u_yy = np.gradient(u_y, dy, axis=0)   # ∂²u/∂y²

    return u_x, u_y, u_xx, u_yy
```

`numpy.gradient` implements central differences automatically (with special handling at boundaries).

---

## Summary

**Finite differences:**
- Approximate derivatives from discrete data
- Central difference: $u'_i \approx (u_{i+1} - u_{i-1})/(2\Delta x)$ (most accurate)
- Second derivative: $u''_i \approx (u_{i+1} - 2u_i + u_{i-1})/(\Delta x)^2$

**Initial conditions:**
- Specify vorticity $\omega(x, y)$ (what you control)
- Solve $\nabla^2 \psi = \omega$ for stream function
- Extract velocity: $u = -\partial\psi/\partial y$, $v = \partial\psi/\partial x$
- Result: incompressible velocity field

**Your three datasets:**
- Same vorticity shape (Gaussian)
- Different centers: $(π, π)$, $(π/2, π)$, $(π, π/2)$
- Different strengths: 1.0, 1.5, 0.8

The N network will learn the PDE dynamics that generalize across these variations.
