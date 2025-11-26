# Navier-Stokes Equations: Complete Notes

## What is a PDE Solution?

A PDE solution is **a function (or set of functions)** that satisfies the equation at every point in the domain.

For Navier-Stokes, the solution is:

$$u(x, y, t), \quad v(x, y, t), \quad p(x, y, t)$$

These are functions. Plug in any position (x, y) and time t → get a number.

### What Makes It "The Solution"?

When you substitute these functions back into the PDE, **the equation is satisfied everywhere**.

Think of it like algebra:

- Equation: $x^2 - 4 = 0$
- Solution: $x = 2$ (or $x = -2$)
- Verification: plug in → $4 - 4 = 0$ ✓

For PDEs:

- Equation: $\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$
- Solution: some function $u(x,t)$
- Verification: compute all the derivatives, plug in → left side equals right side everywhere

### Numerical Solutions

Numerical solving gives you **discrete samples**, not a formula:

```
u[i,j,n] = velocity at grid point (i,j) at timestep n
```

This approximates the continuous function $u(x,y,t)$.

---

## What Are u and v?

- **u** = horizontal velocity (how fast fluid moves in x direction)
- **v** = vertical velocity (how fast fluid moves in y direction)

Together they form the velocity vector:

$$\mathbf{u} = (u, v)$$

At each point (x, y) and time t, you have a little arrow showing which direction and how fast the fluid is moving there.

### Example

At point (3, 5) at time t = 2:
- u(3, 5, 2) = 4 m/s → moving right
- v(3, 5, 2) = -1 m/s → moving slightly down

The fluid there moves mostly rightward with a slight downward drift.

---

## The Nabla Operator (∇)

Nabla is a **vector of partial derivative operators**:

$$\nabla = \left( \frac{\partial}{\partial x}, \frac{\partial}{\partial y} \right)$$

The name changes depending on how it's used:

| Operation | Notation | Name | Input → Output |
|-----------|----------|------|----------------|
| ∇f | nabla on scalar | **Gradient** | scalar → vector |
| ∇·**u** | nabla dot vector | **Divergence** | vector → scalar |
| ∇×**u** | nabla cross vector | **Curl** | vector → vector |
| ∇²f | nabla squared | **Laplacian** | scalar → scalar |

### Examples

**Gradient** (scalar → vector):
$$\nabla p = \left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}\right)$$

**Divergence** (vector → scalar):
$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$$

**Curl** (vector → vector, scalar in 2D):
$$\nabla \times \mathbf{u} = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

**Laplacian** (scalar → scalar):
$$\nabla^2 p = \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2}$$

> [!tip] Key Insight
> Divergence uses **same-direction derivatives** (u with x, v with y) because it's a dot product between nabla and the vector.

---

## What is Navier-Stokes?

Navier-Stokes is a **system of two equations** for incompressible flow:

### 1. Momentum Equation
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$

### 2. Incompressibility Constraint
$$\nabla \cdot \mathbf{u} = 0$$

> [!important] Both Together = Navier-Stokes
> The momentum equation governs how velocity evolves. The incompressibility constraint ensures density stays constant. You need both to close the system.

### Variables and Unknowns

| Symbol | Meaning | Type |
|--------|---------|------|
| **u** | Velocity vector = (u, v) | Vector field |
| u | Horizontal velocity component | Scalar field |
| v | Vertical velocity component | Scalar field |
| p | Pressure | Scalar field |
| ν | Viscosity (mu or nu) | Constant |

> [!note] Bold-u vs Full Solution
> - **u** (bold) = velocity vector = (u, v)
> - Full solution = (u, v, p) or (**u**, p)
> 
> Pressure is separate from velocity in notation, but solving Navier-Stokes gives you all three fields.

---

## Expanding the Momentum Equation

The vector equation separates into **two scalar equations** by matching components.

### Breaking Down Each Term

$$\frac{\partial \mathbf{u}}{\partial t} = \left(\frac{\partial u}{\partial t}, \frac{\partial v}{\partial t}\right)$$

$$(\mathbf{u} \cdot \nabla)\mathbf{u} = \left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}, \quad u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y}\right)$$

$$-\nabla p = \left(-\frac{\partial p}{\partial x}, -\frac{\partial p}{\partial y}\right)$$

$$\nu \nabla^2 \mathbf{u} = \left(\nu \nabla^2 u, \quad \nu \nabla^2 v\right)$$

### The Two Scalar Equations

**x-component (u equation):**
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

**y-component (v equation):**
$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

**Plus incompressibility:**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

> [!summary] Three scalar equations, three unknowns: u, v, p

---

## What Do These Equations Output?

The equations output **acceleration** — the rate of change of velocity.

Isolating the time derivative in the x-equation (move convective terms to right side):

$$\frac{\partial u}{\partial t} = -u\frac{\partial u}{\partial x} - v\frac{\partial u}{\partial y} -\frac{\partial p}{\partial x} + \nu \nabla^2 u$$

The right side gives you: **how fast u is changing at this moment**.

In numerical solving, you use this to update velocity:

$$u^{n+1} = u^n + \Delta t \cdot \frac{\partial u}{\partial t}$$

### The Material Derivative

The full acceleration a fluid particle experiences is:

$$\frac{Du}{Dt} = \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}$$

This is called the **material derivative**. It has two parts:

| Term | Name | Meaning |
|------|------|---------|
| ∂u/∂t | Local acceleration | Velocity changing at a fixed point |
| u·∂u/∂x + v·∂u/∂y | Convective acceleration | Acceleration from moving to a different location |

### Navier-Stokes is Newton's Second Law

Rewritten using the material derivative:

$$\underbrace{\frac{D\mathbf{u}}{Dt}}_{\text{total acceleration}} = \underbrace{-\nabla p + \nu \nabla^2 \mathbf{u}}_{\text{forces per unit mass}}$$

This is just **F = ma** (or a = F/m) for fluids:
- Left side: acceleration of fluid particle
- Right side: forces acting on it (pressure + viscosity)

---

## The Incompressibility Constraint

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

### Physical Meaning

Imagine a tiny box of fluid:

- **∂u/∂x**: How horizontal velocity changes horizontally
  - Positive → box stretches horizontally
- **∂v/∂y**: How vertical velocity changes vertically
  - Positive → box stretches vertically

**Incompressibility means volume can't change.** Stretch in one direction must compress in another.

### Concrete Example: Incompressible Flow

$$u = x, \quad v = -y$$

$$\frac{\partial u}{\partial x} = 1, \quad \frac{\partial v}{\partial y} = -1$$

$$\nabla \cdot \mathbf{u} = 1 + (-1) = 0 \checkmark$$

Stretches horizontally, compresses vertically → volume preserved.

### Counterexample: Compressible Flow

$$u = x, \quad v = y$$

$$\nabla \cdot \mathbf{u} = 1 + 1 = 2 \neq 0$$

Expanding in both directions → density would drop → violates incompressibility.

---

## Rotation vs Incompressibility

These are **independent properties**:

| Property | Operator | What it measures |
|----------|----------|------------------|
| Incompressibility | ∇·**u** = ∂u/∂x + ∂v/∂y | Expansion/compression |
| Rotation (vorticity) | ∇×**u** = ∂v/∂x − ∂u/∂y | Spinning motion |

### Example: Pure Rotation

$$u = -y, \quad v = x$$

**Divergence:** ∂u/∂x + ∂v/∂y = 0 + 0 = 0 ✓ (incompressible)

**Vorticity:** ∂v/∂x − ∂u/∂y = 1 − (−1) = 2 (rotating)

> [!note] Key Point
> A flow can be:
> - Incompressible and rotating
> - Incompressible and non-rotating
> - Compressible and rotating
> - Compressible and non-rotating

---

## Finite Difference Method

### Core Idea

Replace continuous derivatives with algebraic approximations using grid values.

### Derivative Approximations

**First derivative (centered):**
$$\frac{du}{dx} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}$$

**Second derivative:**
$$\frac{d^2u}{dx^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$

### Example: Heat Equation

PDE: $\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial x^2}$

Discretized:
$$\frac{u_i^{n+1} - u_i^n}{\Delta t} = \nu \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\Delta x^2}$$

Update rule:
$$u_i^{n+1} = u_i^n + \frac{\nu \Delta t}{\Delta x^2}(u_{i+1}^n - 2u_i^n + u_{i-1}^n)$$

Start with initial condition, march forward in time.

---

## The Vorticity-Streamfunction Method

### The Problem with Direct Velocity Formulation

Pressure p has no evolution equation. It's implicitly defined as whatever enforces ∇·**u** = 0. This requires an extra Poisson solve each timestep.

### The Loophole: Eliminate Pressure

Take the **curl** of the momentum equation. Since curl(∇p) = 0:

$$\frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla)\omega = \nu \nabla^2 \omega$$

where **vorticity** is:
$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

No pressure term! We can evolve ω directly.

### Recovering Velocity via Streamfunction

Define streamfunction ψ such that:
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

> [!tip] This Automatically Satisfies Incompressibility
> $$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = \frac{\partial^2 \psi}{\partial x \partial y} - \frac{\partial^2 \psi}{\partial y \partial x} = 0$$

Vorticity relates to streamfunction by:
$$\omega = \nabla^2 \psi$$

### The Algorithm

```
For each timestep:
    1. Evolve vorticity ω using vorticity equation
    2. Solve Poisson equation: ∇²ψ = ω
    3. Extract velocity: u = ∂ψ/∂y, v = −∂ψ/∂x
```

---

## Recovering Pressure (If Needed)

Often you only need velocity. But if pressure is required:

### Derivation

Take divergence of momentum equation:

$$\nabla \cdot \left[\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}\right]$$

- Time derivative term → 0 (incompressibility)
- Viscous term → 0 (incompressibility)
- Pressure term → −∇²p

What remains:
$$\nabla^2 p = -\nabla \cdot \left[(\mathbf{u} \cdot \nabla)\mathbf{u}\right]$$

### Steps to Recover p

1. You already have u, v from vorticity-streamfunction
2. Compute RHS using known velocity derivatives
3. Solve Poisson equation ∇²p = RHS
4. Apply boundary conditions (usually Neumann from momentum equation)

---

## Why Vorticity-Streamfunction for 2D?

### Advantages

1. **Eliminates pressure** — no extra solve needed
2. **Reduces unknowns** — track just ω instead of (u, v, p)
3. **Incompressibility automatic** — streamfunction guarantees it

### Limitations

**Doesn't generalize well to 3D:**
- Vorticity becomes a vector (ω_x, ω_y, ω_z)
- No simple streamfunction equivalent
- Other methods (projection, pressure-Poisson) preferred

### Alternative Methods

| Method | Description |
|--------|-------------|
| **Projection/Fractional-step** | Solve momentum ignoring incompressibility → project onto divergence-free space |
| **Pressure-Poisson** | Keep velocity variables, solve Poisson for pressure each timestep |

> [!note] For Your Project
> Vorticity-streamfunction is convenient for 2D Navier-Stokes. But projection methods work too if you need velocity/pressure directly.

---

## Quick Reference

### The Full System (2D Incompressible Navier-Stokes)

**Vector form:**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$
$$\nabla \cdot \mathbf{u} = 0$$

**Vorticity form:**
$$\frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla)\omega = \nu \nabla^2 \omega$$
$$\nabla^2 \psi = \omega$$
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

### Key Relationships

| Quantity | Definition |
|----------|------------|
| Vorticity | ω = ∂v/∂x − ∂u/∂y |
| Divergence | ∇·**u** = ∂u/∂x + ∂v/∂y |
| Streamfunction | u = ∂ψ/∂y, v = −∂ψ/∂x |
| Poisson for ψ | ∇²ψ = ω |
| Poisson for p | ∇²p = −∇·[(**u**·∇)**u**] |
