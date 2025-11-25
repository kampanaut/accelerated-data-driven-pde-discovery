# Project Plan Amendments Log

**Purpose:** Track all identified corrections, clarifications, and amendments to the submitted project plan throughout the project lifecycle.

**Last Updated:** November 18, 2025

---

## Amendment 1: Mathematical Relationship Between Burgers and Navier-Stokes

**Source:** Supervisor feedback (November 4, 2025 meeting)  
**Status:** REQUIRED - Not yet addressed  
**Priority:** HIGH (described as "potentially ruinous oversight")

### Issue
The literature survey references Burgers and Navier-Stokes as "related" equations without providing explicit mathematical explanation of their relationship. This omission prevents:
- Justifying why cross-domain transfer should work
- Understanding what structural features enable/prevent transfer
- Defending Advanced deliverable design choices

### Required Addition
Add explicit mathematical derivation section showing:

1. **Navier-Stokes equations** (2D incompressible):
   - Momentum: $\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$
   - Continuity: $\nabla \cdot \mathbf{u} = 0$

2. **Burgers equation** (1D):
   - $\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$

3. **Shared mathematical structure:**
   - Nonlinear advection terms: $u \cdot u_x$ vs $(\mathbf{u} \cdot \nabla)\mathbf{u}$
   - Viscous diffusion: $\nu \cdot u_{xx}$ vs $\nu \nabla^2 \mathbf{u}$
   - Same viscosity parameter $\nu$ (target for identification)

4. **Key differences:**
   - Pressure coupling and incompressibility constraint in N-S
   - Vector-valued vs scalar solutions
   - Dimensionality (2D/3D vs 1D)
   - Divergence of dyadic product vs simple quadratic nonlinearity

5. **Transfer hypothesis:**
   - Explicit argument for why shared advection-diffusion structure might enable cross-domain transfer
   - Acknowledge uncertainty: "Whether this shared structure is sufficient for meta-learning transfer remains an open question"

### Location in Document
Add new subsection in Literature Survey between "Data-Driven Methods for Chaotic Systems" and "Deep Neural Network Architectures" sections.

---

## Amendment 2: Chaos Terminology and Framing

**Source:** Supervisor feedback + internal discussion  
**Status:** REQUIRED - Not yet addressed  
**Priority:** MEDIUM

### Issue
Project plan includes Burgers equation in "Data-Driven Methods for Chaotic Systems" section, but Burgers equation does not exhibit chaos. This creates conceptual confusion about what the project studies.

### Required Changes

**Option A (Conservative):**
- Reframe section title from "Chaotic Systems" to "Nonlinear Dynamical Systems"
- Remove chaos-specific language when discussing Burgers
- Clarify: Burgers is deterministic and predictable; Navier-Stokes CAN be chaotic at high Reynolds numbers
- Position project as studying "nonlinear dynamics with varying complexity" rather than chaos specifically

**Option B (More ambitious):**
- Investigate whether Burgers exhibits complex/chaotic behavior in certain parameter regimes
- If not, acknowledge this as limitation and focus on "complexity gradient" from simple (Burgers) to potentially chaotic (N-S)

### Affected Sections
- Key Terminology (definition of "Chaotic Systems")
- Data-Driven Methods for Chaotic Systems (entire section title and content)
- Project Description (first paragraph mentions chaos)

---

## Amendment 3: I1 Deliverable Modification

**Source:** November 4, 2025 supervision meeting  
**Status:** ALREADY ADDRESSED in meeting  
**Priority:** COMPLETE

### Change Made
**Original I1:** "Spectral Methods & Comparative Analysis: Implement Fourier/Chebyshev spectral methods; compare accuracy and convergence vs. finite difference approach"

**New I1:** "Noise Effects Study: Systematically measure impact of noise in training data on meta-learning performance and PDE discovery accuracy"

### Rationale
- Original deliverable was "at-odds with the remainder of the project" (supervisor feedback)
- New deliverable focuses on ML aspects rather than numerical methods comparison
- Directly relevant to robustness of meta-learning approach
- Tests the proposed 1% Gaussian noise addition mentioned in project description

**Note:** This change is already reflected in practice but should be formalized in any revised project plan document.

---

## Amendment 4: Timeline Feasibility

**Source:** Supervisor feedback  
**Status:** ACKNOWLEDGED - Detailed revision pending  
**Priority:** MEDIUM

### Issue
Supervisor: "Based on the day counts for some of these, I think this timeline is very ambitious. Several things will take much longer to understand and implement than a single week."

### Specific Concerns
- B1 (Data Generation & Validation): 27 days â†’ likely underestimated for debugging numerical solvers
- B3 (Meta-Learning Implementation): 14 days â†’ MAML implementation, hyperparameter tuning, evaluation typically takes longer
- I1 (now Noise Effects Study): 14 days â†’ systematic noise study with multiple levels requires significant experimentation
- I2 (Navier-Stokes Extension): 21 days â†’ 2D spectral methods and validation is complex

### Recommended Approach
- Create visual timeline (Gantt chart) showing task dependencies
- Identify critical path and sequential dependencies
- Add buffer time for debugging and iteration
- Consider which deliverables can proceed in parallel
- Be realistic about learning curve for new concepts (MAML, spectral methods, N-S solvers)

### Action Item
Revise timeline with more conservative estimates after B1 milestone completion provides calibration data.

---

## Amendment 5: Mathematical Depth in Literature Survey

**Source:** Supervisor feedback  
**Status:** REQUIRED - Not yet addressed  
**Priority:** MEDIUM-HIGH

### Issue
Supervisor: "There is a lack of mathematics in the presentation; this gives the impression of only engaging with the literature superficially."

### Required Changes
Add mathematical details to key papers:

**PINNs section:**
- Include loss function formulation: $\text{MSE} = \text{MSE}_u + \text{MSE}_f$
- Show physics loss explicitly: $\text{MSE}_f = \frac{1}{N_f}\sum_{i=1}^{N_f} |f(x_i)|^2$ where $f$ is PDE residual
- Explain automatic differentiation for computing derivatives

**MAML section:**
- Show bi-level optimization explicitly
- Inner loop: $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$
- Outer loop: $\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}(\theta_i')$

**ANIL section:**
- Specify which parameters freeze vs adapt
- Show reduced computation cost mathematically

### General Principle
For each cited work, include:
- Key equations or formulations
- Mathematical notation explained
- How the mathematics relates to project approach

---

## Amendment 6: Critical Engagement with Literature

**Source:** Supervisor feedback  
**Status:** REQUIRED - Not yet addressed  
**Priority:** MEDIUM

### Issue
Supervisor: "The rest of the literature on PINNs is summarised with a single reference; no critical engagement with the limitations or successes of the methods is demonstrated."

### Required Additions
For each major method/paper discussed:

**What worked:**
- Specific problems solved successfully
- Quantitative results (accuracy, speed)
- Conditions under which method excels

**Limitations:**
- Known failure modes
- Computational costs
- Data requirements
- Theoretical gaps

**Relevance to project:**
- How these limitations affect your approach
- Why certain design choices avoid known problems
- What tradeoffs you're making

### Example Application
For PINNs:
- Success: Can solve forward problems with sparse data
- Limitation: Training instability, need careful hyperparameter tuning, difficulty with high-frequency solutions
- Relevance: Your decoupled approach might improve stability by separating solution and operator learning

---

## Amendment 7: Parametrization Strategy Documentation

**Source:** November 4, 2025 meeting + November 18 discussion  
**Status:** IN PROGRESS  
**Priority:** HIGH (immediate need for B1 deliverable)

### Issue
Project plan does not specify how task distributions will be constructed. This is fundamental to meta-learning approach.

### Required Specification

**Burgers Equation Task Distribution:**
- Primary parameter: Viscosity $\nu$
- Sampling strategy: [TO BE DETERMINED - uniform vs logarithmic vs random]
- Range: $\nu \in [0.01, 0.1]$ (tentative)
- Number of tasks: 20-30 (for meta-training)
- Fixed parameters:
  - Initial condition: $u(x,0) = -\sin(x)$
  - Boundary conditions: Periodic on $x \in [0, 2\pi]$
  - Time range: $t \in [0, 1]$
  - Spatial resolution: 256 points (tentative)

**Navier-Stokes Task Distribution:**
- Primary parameter: Viscosity $\nu$
- Sampling strategy: [TO BE DETERMINED]
- Range: $\nu \in [0.001, 0.01]$ (tentative)
- Number of tasks: 15-20
- Fixed parameters:
  - Initial condition: Taylor-Green vortex
  - Boundary conditions: Doubly periodic on $[0, 2\pi] \times [0, 2\pi]$
  - Spatial resolution: $128 \times 128$ (tentative)

### Open Questions
1. Uniform vs logarithmic sampling of $\nu$?
2. Should we vary other parameters (ICs, BCs) or keep them fixed?
3. Optimal number of tasks for meta-learning?
4. Resolution requirements for accurate derivatives?

### Next Steps
- Explore $\nu$ values using VisualPDE
- Determine what range gives interesting behavior variation
- Finalize sampling strategy before implementing B1

---

## Amendment 8: Noise Study Specification (I1 Deliverable)

**Source:** November 4, 2025 meeting  
**Status:** PLANNED - Details needed  
**Priority:** MEDIUM (becomes HIGH after B3 completion)

### Required Specification
Since I1 deliverable changed to noise effects study, need to define:

**Noise levels to test:**
- Baseline: 0% noise (clean derivatives)
- Low: 1% Gaussian noise (as mentioned in project description)
- Medium: 5% Gaussian noise
- High: 10% Gaussian noise

**Noise application:**
- Add to derivatives: $(u_t, u_x, u_{xx})$ after computation
- Gaussian: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ where $\sigma$ = noise_level Ã— signal_magnitude

**Evaluation metrics:**
- Parameter identification accuracy (|$\nu_{true} - \nu_{predicted}$|)
- Convergence speed (gradient steps to threshold accuracy)
- Robustness across task distribution

**Experimental design:**
- Train meta-learned Network 2 with each noise level
- Test on held-out tasks with matching noise
- Test on held-out tasks with mismatched noise (train on 1%, test on 5%)

---

## Future Amendments Section

**Placeholder for additional amendments identified in subsequent conversations.**

### To Be Added:
- Clarifications on evaluation methodology
- Baseline comparison specifications
- Cross-domain transfer experimental design details
- Any timeline adjustments based on B1 progress
- Computational resource requirements
- Specific software libraries and tools

---

## Notes on Document Usage

**Purpose:** This document tracks "amendments" conceptually but does NOT replace the official submitted project plan. These are notes on what would be changed if revising, and serve as:

1. **Reference for future conversations** - Context on what's been discussed
2. **Preparation for presentations/demos** - Know what questions to anticipate
3. **Final paper writing** - Ensure all identified issues are addressed in methodology
4. **Posterity** - Historical record of how project evolved

**Update Protocol:**
- Add new amendments as they're identified in meetings or discussions
- Update status as amendments are addressed in practice
- Date all major updates

**Related Documents:**
- Project Plan (official submission) - unchanged
- Meeting logs - capture discussions leading to amendments
- This amendments log - tracks what should change
