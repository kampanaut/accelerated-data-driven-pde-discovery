"""Generate all figures for docs/explainers/spiral_waves_two_models.md.

Uses analytical/synthetic spiral wave data — no PDE solver needed.
Outputs 8 PNGs to docs/explainers/figures/sw_*.png.

Usage:
    uv run python scripts/generate_spiral_figures.py
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUT_DIR = Path("docs/explainers/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def archimedean_phase(x: np.ndarray, y: np.ndarray,
                      cx: float, cy: float, k: float) -> np.ndarray:
    """Spiral phase field θ = atan2(y-cy, x-cx) - k·r."""
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx**2 + dy**2)
    return np.arctan2(dy, dx) - k * r


def spectral_laplacian_2d(field: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """Compute ∇²field via FFT on a periodic domain [0,Lx]×[0,Ly]."""
    ny, nx = field.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky)
    f_hat = np.fft.fft2(field)
    lap_hat = -(KX**2 + KY**2) * f_hat
    return np.fft.ifft2(lap_hat).real


# ---------------------------------------------------------------------------
# Figure 1: Spiral wave concept (circular → broken → spiral)
# ---------------------------------------------------------------------------

def fig01_spiral_concept():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    N = 512
    L = 10.0
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Panel 1: concentric circular waves
    wave1 = np.cos(2.0 * R)
    axes[0].imshow(wave1, extent=[-L, L, -L, L], cmap="RdBu_r",
                   vmin=-1, vmax=1, origin="lower")
    axes[0].set_title("Circular wave", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Panel 2: broken wave (mask a sector to simulate break)
    wave2 = np.cos(2.0 * R).copy()
    angle = np.arctan2(Y, X)
    mask = (angle > 0) & (angle < np.pi / 3) & (R > 2) & (R < 5)
    wave2[mask] = 0.0
    axes[1].imshow(wave2, extent=[-L, L, -L, L], cmap="RdBu_r",
                   vmin=-1, vmax=1, origin="lower")
    axes[1].set_title("Broken wave", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    # Arrow pointing to the break
    axes[1].annotate("free end", xy=(2.5, 3.0), fontsize=10,
                     color="black", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                     xytext=(5, 7))

    # Panel 3: spiral wave (Archimedean)
    theta = archimedean_phase(X, Y, 0, 0, k=1.2)
    spiral = np.cos(theta)
    # Fade at center
    fade = np.tanh(R / 1.5)
    spiral *= fade
    axes[2].imshow(spiral, extent=[-L, L, -L, L], cmap="RdBu_r",
                   vmin=-1, vmax=1, origin="lower")
    axes[2].set_title("Spiral wave", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    # Mark the core
    axes[2].plot(0, 0, "k*", markersize=10)
    axes[2].annotate("core", xy=(0, 0), xytext=(2, 3), fontsize=10,
                     fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    for ax in axes:
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_01_spiral_wave_concept.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_01 done")


# ---------------------------------------------------------------------------
# Figure 2: Excitable vs oscillatory media
# ---------------------------------------------------------------------------

def fig02_excitable_vs_oscillatory():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    N = 256
    L = 10.0
    x = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    # --- Top row: excitable (sharp wavefront propagating outward) ---
    for col, t_offset in enumerate([0.0, 2.5, 5.0]):
        theta = archimedean_phase(X, Y, 0, 0, k=1.5)
        # Sharp excitable pulse: narrow band where cos(θ + offset) crosses zero
        phase = theta + t_offset
        # Wavefront: sharp sigmoid of cos(phase)
        u = 0.5 * (1.0 + np.tanh(np.cos(phase) / 0.08))
        # Refractory tail: suppress behind wavefront
        u *= np.tanh(R / 1.5)  # fade at core
        axes[0, col].imshow(u, extent=[-L, L, -L, L], cmap="hot",
                            vmin=0, vmax=1, origin="lower")
        axes[0, col].set_title(f"Excitable  t={col}", fontsize=12,
                               fontweight="bold")
        axes[0, col].set_aspect("equal")
        if col == 0:
            axes[0, col].set_ylabel("y", fontsize=11)

    # --- Bottom row: oscillatory (smooth phase waves rotating) ---
    for col, t_offset in enumerate([0.0, 1.0, 2.0]):
        theta = archimedean_phase(X, Y, 0, 0, k=1.0)
        phase = theta + t_offset
        R_amp = np.sqrt(1.0) * np.tanh(R / 1.0)
        u = R_amp * np.cos(phase)
        v = R_amp * np.sin(phase)
        # Show as phase colorwheel: hue = phase, brightness = amplitude
        phase_norm = (np.arctan2(v, u) + np.pi) / (2 * np.pi)
        amp = np.sqrt(u**2 + v**2)
        amp_norm = amp / amp.max()
        hsv = np.stack([phase_norm, np.ones_like(phase_norm),
                        amp_norm], axis=-1)
        rgb = mcolors.hsv_to_rgb(hsv)
        axes[1, col].imshow(rgb, extent=[-L, L, -L, L], origin="lower")
        axes[1, col].set_title(f"Oscillatory  t={col}", fontsize=12,
                               fontweight="bold")
        axes[1, col].set_aspect("equal")
        if col == 0:
            axes[1, col].set_ylabel("y", fontsize=11)

    # Row labels
    fig.text(0.02, 0.73, "Sharp\nwavefronts", fontsize=11, fontweight="bold",
             va="center", rotation=0, color="#CC3300")
    fig.text(0.02, 0.28, "Smooth\nphase waves", fontsize=11, fontweight="bold",
             va="center", rotation=0, color="#3366CC")

    fig.tight_layout(rect=(0.06, 0.0, 1.0, 1.0))
    fig.savefig(OUT_DIR / "sw_02_excitable_vs_oscillatory.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_02 done")


# ---------------------------------------------------------------------------
# Figure 3: FHN phase portrait
# ---------------------------------------------------------------------------

def fig03_fhn_phase_portrait():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    a, b, eps = 0.7, 0.8, 0.08

    # Nullclines
    u_vals = np.linspace(-2.5, 2.5, 500)
    v_nullcline_u = u_vals - u_vals**3 / 3        # u-nullcline: v = u - u³/3
    v_nullcline_v = (u_vals + a) / b               # v-nullcline: v = (u+a)/b

    ax.plot(u_vals, v_nullcline_u, "b-", lw=2.5, label=r"$u$-nullcline: $v = u - u^3/3$")
    ax.plot(u_vals, v_nullcline_v, "r-", lw=2.5, label=r"$v$-nullcline: $v = (u+a)/b$")

    # Integrate the ODE to get a trajectory (kicked from rest)
    def fhn_ode(_t: float, state: list[float]) -> list[float]:
        u, v = state
        du = u - u**3 / 3 - v
        dv = eps * (u + a - b * v)
        return [du, dv]

    # Find resting state (intersection)
    # Approximate: solve u - u³/3 = (u+a)/b
    from scipy.optimize import fsolve
    def eq(u):
        return u - u**3 / 3 - (u + a) / b
    u_rest = fsolve(eq, -1.2)[0]
    v_rest = (u_rest + a) / b

    # Kick the system past threshold
    u0_kick = u_rest + 1.0
    v0_kick = v_rest

    sol = solve_ivp(fhn_ode, [0, 200], [u0_kick, v0_kick],
                    max_step=0.1, rtol=1e-8, atol=1e-10)

    ax.plot(sol.y[0], sol.y[1], "k-", lw=1.2, alpha=0.7, zorder=3)

    # Add arrows along trajectory
    arrow_indices = [50, 300, 700, 1200]
    for idx in arrow_indices:
        if idx + 5 < len(sol.y[0]):
            ax.annotate("", xy=(sol.y[0][idx + 5], sol.y[1][idx + 5]),
                        xytext=(sol.y[0][idx], sol.y[1][idx]),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=1.5))

    # Mark resting state
    ax.plot(u_rest, v_rest, "go", markersize=10, zorder=5)
    ax.annotate("resting\nstate", xy=(u_rest, v_rest),
                xytext=(u_rest - 1.0, v_rest + 0.5), fontsize=10,
                fontweight="bold", color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))

    # Mark kick
    ax.plot(u0_kick, v0_kick, "r^", markersize=10, zorder=5)
    ax.annotate("stimulus", xy=(u0_kick, v0_kick),
                xytext=(u0_kick + 0.3, v0_kick + 0.5), fontsize=10,
                fontweight="bold", color="red")

    # Labels for phases of the action potential
    ax.annotate("1. excitation\n   (fast →)", xy=(1.5, -0.5),
                fontsize=9, color="#555555", fontstyle="italic")
    ax.annotate("2. recovery\n   (slow ↑)", xy=(1.8, 1.0),
                fontsize=9, color="#555555", fontstyle="italic")
    ax.annotate("3. de-excitation\n   (fast ←)", xy=(-2.2, 1.8),
                fontsize=9, color="#555555", fontstyle="italic")
    ax.annotate("4. return\n   (slow ↓)", xy=(-1.8, -0.2),
                fontsize=9, color="#555555", fontstyle="italic")

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.0, 2.5)
    ax.set_xlabel("u (activator / voltage)", fontsize=12)
    ax.set_ylabel("v (inhibitor / recovery)", fontsize=12)
    ax.set_title("FitzHugh-Nagumo Phase Portrait", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_03_fhn_phase_portrait.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_03 done")


# ---------------------------------------------------------------------------
# Figure 4: Synthetic FHN spiral
# ---------------------------------------------------------------------------

def make_fhn_spiral(N: int = 512, L: float = 20.0, k: float = 0.8,
                    width: float = 0.06, n_arms: int = 1):
    """Construct a synthetic FHN-like spiral with sharp wavefronts.

    Returns x, y meshgrid arrays and (u, v) fields.
    """
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    cx, cy = L / 2, L / 2

    theta = archimedean_phase(X, Y, cx, cy, k=k)
    theta *= n_arms
    R_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Sharp sigmoid wavefront for u
    u = 0.5 * (1.0 + np.tanh(np.cos(theta) / width))
    # Fade at core
    core_fade = np.tanh(R_dist / 2.0)
    u *= core_fade

    # FHN-like v: slow recovery behind wavefront
    # v peaks where u is declining (quarter-wavelength behind u peak)
    v = 0.5 * (1.0 + np.tanh(np.sin(theta) / (width * 3)))
    v *= core_fade * 0.6  # smaller amplitude

    return X, Y, u, v


def fig04_fhn_spiral():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _, _, u, v = make_fhn_spiral()
    L = 20.0

    im0 = axes[0].imshow(u, extent=[0, L, 0, L], cmap="hot",
                         vmin=0, vmax=1, origin="lower")
    axes[0].set_title("$u$ (activator)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(v, extent=[0, L, 0, L], cmap="YlGnBu",
                         vmin=0, vmax=0.6, origin="lower")
    axes[1].set_title("$v$ (inhibitor / recovery)", fontsize=13,
                      fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Annotate
    axes[0].annotate("sharp\nwavefront", xy=(12, 14), fontsize=10,
                     color="white", fontweight="bold",
                     ha="center")
    axes[0].annotate("resting\nregion", xy=(5, 5), fontsize=10,
                     color="yellow", fontweight="bold",
                     ha="center")
    axes[0].plot(10, 10, "w*", markersize=8)
    axes[0].annotate("core", xy=(10, 10), xytext=(7, 7),
                     fontsize=10, color="white", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    fig.suptitle("Synthetic FHN-like Spiral Wave", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_04_fhn_spiral.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_04 done")


# ---------------------------------------------------------------------------
# Figure 5: Lambda-omega polar dynamics
# ---------------------------------------------------------------------------

def fig05_lambda_omega_polar():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    a_param = 1.0
    c_param = 2.0

    # --- Left panel: limit cycle in (u, v) plane ---
    theta_lc = np.linspace(0, 2 * np.pi, 200)
    R_lc = np.sqrt(a_param)
    u_lc = R_lc * np.cos(theta_lc)
    v_lc = R_lc * np.sin(theta_lc)

    axes[0].plot(u_lc, v_lc, "b-", lw=2.5, label=f"limit cycle $R = \\sqrt{{a}} = {R_lc:.1f}$")
    axes[0].plot(0, 0, "ko", markersize=6)
    axes[0].annotate("unstable\nfixed point", xy=(0, 0), xytext=(0.3, -0.5),
                     fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))

    # Arrows showing direction of orbit
    for i_arrow in [25, 75, 125, 175]:
        axes[0].annotate("",
                         xy=(u_lc[i_arrow + 2], v_lc[i_arrow + 2]),
                         xytext=(u_lc[i_arrow], v_lc[i_arrow]),
                         arrowprops=dict(arrowstyle="-|>", color="blue",
                                         lw=2, mutation_scale=15))

    # Spiraling-in trajectory from outside
    t_in = np.linspace(0, 30, 1000)
    R_in = R_lc + 0.5 * np.exp(-0.15 * t_in)
    u_in = R_in * np.cos(c_param * t_in)
    v_in = R_in * np.sin(c_param * t_in)
    axes[0].plot(u_in, v_in, "gray", lw=0.8, alpha=0.5)

    # Spiraling-out from inside
    R_out = R_lc - 0.7 * np.exp(-0.15 * t_in)
    R_out = np.maximum(R_out, 0.01)
    u_out = R_out * np.cos(c_param * t_in)
    v_out = R_out * np.sin(c_param * t_in)
    axes[0].plot(u_out, v_out, "gray", lw=0.8, alpha=0.5)

    axes[0].set_xlim(-1.8, 1.8)
    axes[0].set_ylim(-1.8, 1.8)
    axes[0].set_xlabel("$u$", fontsize=12)
    axes[0].set_ylabel("$v$", fontsize=12)
    axes[0].set_title("Limit Cycle", fontsize=13, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # --- Middle panel: R dynamics (dR/dt vs R) ---
    R_vals = np.linspace(0, 2.0, 200)
    dRdt = R_vals * (a_param - R_vals**2)

    axes[1].plot(R_vals, dRdt, "b-", lw=2.5)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].axvline(R_lc, color="green", lw=1.5, ls="--", alpha=0.7)
    axes[1].plot(R_lc, 0, "go", markersize=10, zorder=5)
    axes[1].annotate(f"$R = \\sqrt{{a}} = {R_lc:.1f}$",
                     xy=(R_lc, 0), xytext=(R_lc + 0.3, 0.15),
                     fontsize=10, color="green", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="green"))

    # Arrows showing flow direction
    axes[1].annotate("", xy=(0.6, 0.02), xytext=(0.3, 0.02),
                     arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    axes[1].annotate("grows", xy=(0.4, 0.08), fontsize=9, color="red")
    axes[1].annotate("", xy=(1.3, -0.02), xytext=(1.6, -0.02),
                     arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    axes[1].annotate("shrinks", xy=(1.35, -0.1), fontsize=9, color="red")

    axes[1].set_xlabel("$R$ (amplitude)", fontsize=12)
    axes[1].set_ylabel("$dR/dt$", fontsize=12)
    axes[1].set_title("Amplitude Dynamics: $\\dot{R} = R(a - R^2)$",
                      fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # --- Right panel: phase vs time ---
    t_phase = np.linspace(0, 10, 200)
    omega = c_param * a_param
    theta_phase = omega * t_phase

    axes[2].plot(t_phase, theta_phase, "b-", lw=2.5)
    axes[2].set_xlabel("Time", fontsize=12)
    axes[2].set_ylabel(r"$\theta$ (phase)", fontsize=12)
    axes[2].set_title(f"Phase Dynamics: $\\dot{{\\theta}} = cR^2 = {omega:.0f}$",
                      fontsize=13, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    # Mark one period
    T = 2 * np.pi / omega
    axes[2].axhline(2 * np.pi, color="gray", lw=0.8, ls=":")
    axes[2].annotate(f"one cycle ($T = 2\\pi/\\omega = {T:.2f}$)",
                     xy=(T, 2 * np.pi), xytext=(T + 1.5, 5),
                     fontsize=9, arrowprops=dict(arrowstyle="->", lw=1))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_05_lambda_omega_polar.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_05 done")


# ---------------------------------------------------------------------------
# Figure 6: Synthetic lambda-omega spiral (phase colorwheel)
# ---------------------------------------------------------------------------

def make_lo_spiral(N: int = 512, L: float = 20.0, a_param: float = 1.0,
                   k: float = 0.8):
    """Construct a synthetic λ-ω spiral.

    Returns X, Y meshgrid and (u, v) fields.
    """
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    cx, cy = L / 2, L / 2

    R_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    theta = archimedean_phase(X, Y, cx, cy, k=k)

    R_amp = np.sqrt(a_param) * np.tanh(R_dist / 2.0)
    u = R_amp * np.cos(theta)
    v = R_amp * np.sin(theta)

    return X, Y, u, v


def fig06_lambda_omega_spiral():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    N, L = 512, 20.0
    _, _, u, v = make_lo_spiral(N=N, L=L)

    # Panel 1: u field
    im0 = axes[0].imshow(u, extent=[0, L, 0, L], cmap="RdBu_r",
                         vmin=-1.1, vmax=1.1, origin="lower")
    axes[0].set_title("$u$ field", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Panel 2: v field
    im1 = axes[1].imshow(v, extent=[0, L, 0, L], cmap="RdBu_r",
                         vmin=-1.1, vmax=1.1, origin="lower")
    axes[1].set_title("$v$ field", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Panel 3: phase colorwheel
    phase = np.arctan2(v, u)
    amp = np.sqrt(u**2 + v**2)
    phase_norm = (phase + np.pi) / (2 * np.pi)
    amp_norm = amp / amp.max()
    hsv = np.stack([phase_norm, np.ones_like(phase_norm), amp_norm],
                   axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    axes[2].imshow(rgb, extent=[0, L, 0, L], origin="lower")
    axes[2].set_title("Phase (hue) + Amplitude (brightness)",
                      fontsize=13, fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_aspect("equal")

    # Mark core
    for ax in axes:
        ax.plot(10, 10, "k*", markersize=8)

    fig.suptitle("Synthetic $\\lambda$-$\\omega$ Spiral Wave",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_06_lambda_omega_spiral.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_06 done")


# ---------------------------------------------------------------------------
# Figure 7: u vs ∇²u scatter (Brusselator, FHN, λ-ω)
# ---------------------------------------------------------------------------

def fig07_u_vs_laplacian():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    N = 256
    L = 20.0

    # --- Brusselator (synthetic Turing pattern) ---
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    kc = 2 * np.pi * 3 / L  # dominant wavenumber (3 wavelengths across domain)
    u_br = 3.0 + 0.5 * np.cos(kc * X) * np.cos(kc * Y)
    # Add a tiny second mode so it's not perfectly degenerate
    u_br += 0.02 * np.cos(2 * kc * X)
    lap_br = spectral_laplacian_2d(u_br, L, L)

    u_flat = u_br.ravel()
    lap_flat = lap_br.ravel()
    # Subsample for scatter
    idx = np.random.default_rng(42).choice(len(u_flat), 5000, replace=False)
    corr_br = np.corrcoef(u_flat[idx], lap_flat[idx])[0, 1]

    axes[0].scatter(u_flat[idx], lap_flat[idx], s=1, alpha=0.3, c="C0")
    axes[0].set_title(f"Brusselator (Turing)\n$r = {corr_br:.3f}$",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("$u$", fontsize=11)
    axes[0].set_ylabel("$\\nabla^2 u$", fontsize=11)

    # --- FHN spiral ---
    _, _, u_fhn, _v_fhn = make_fhn_spiral(N=N, L=L)
    lap_fhn = spectral_laplacian_2d(u_fhn, L, L)

    u_flat_f = u_fhn.ravel()
    lap_flat_f = lap_fhn.ravel()
    idx_f = np.random.default_rng(42).choice(len(u_flat_f), 5000, replace=False)
    corr_fhn = np.corrcoef(u_flat_f[idx_f], lap_flat_f[idx_f])[0, 1]

    axes[1].scatter(u_flat_f[idx_f], lap_flat_f[idx_f], s=1, alpha=0.3,
                    c="C1")
    axes[1].set_title(f"FHN (Excitable spiral)\n$r = {corr_fhn:.3f}$",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("$u$", fontsize=11)
    axes[1].set_ylabel("$\\nabla^2 u$", fontsize=11)

    # --- Lambda-omega spiral ---
    _, _, u_lo, _v_lo = make_lo_spiral(N=N, L=L)
    lap_lo = spectral_laplacian_2d(u_lo, L, L)

    u_flat_l = u_lo.ravel()
    lap_flat_l = lap_lo.ravel()
    idx_l = np.random.default_rng(42).choice(len(u_flat_l), 5000, replace=False)
    corr_lo = np.corrcoef(u_flat_l[idx_l], lap_flat_l[idx_l])[0, 1]

    axes[2].scatter(u_flat_l[idx_l], lap_flat_l[idx_l], s=1, alpha=0.3,
                    c="C2")
    axes[2].set_title(f"$\\lambda$-$\\omega$ (Oscillatory spiral)\n$r = {corr_lo:.3f}$",
                      fontsize=12, fontweight="bold")
    axes[2].set_xlabel("$u$", fontsize=11)
    axes[2].set_ylabel("$\\nabla^2 u$", fontsize=11)

    # Annotations
    axes[0].text(0.05, 0.92, "COLLINEAR\n(line)", transform=axes[0].transAxes,
                 fontsize=10, fontweight="bold", color="red", va="top")
    axes[1].text(0.05, 0.92, "CLOUD\n(independent)", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold", color="green", va="top")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle("$u$ vs $\\nabla^2 u$ — Collinearity Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_07_fhn_u_vs_laplacian.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_07 done")


# ---------------------------------------------------------------------------
# Figure 8: Amplitude comparison (R = sqrt(u² + v²))
# ---------------------------------------------------------------------------

def fig08_amplitude_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    N = 256
    L = 20.0
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x)

    # --- Brusselator (Turing): nearly uniform amplitude ---
    kc = 2 * np.pi * 3 / L
    u_br = 3.0 + 0.5 * np.cos(kc * X) * np.cos(kc * Y)
    v_br = 1.5 + 0.3 * np.cos(kc * X) * np.cos(kc * Y + np.pi / 4)
    R_br = np.sqrt(u_br**2 + v_br**2)

    im0 = axes[0].imshow(R_br, extent=[0, L, 0, L], cmap="viridis",
                         origin="lower")
    axes[0].set_title("Brusselator\n$R$ nearly uniform", fontsize=12,
                      fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046)
    cb0.set_label("$R = \\sqrt{u^2 + v^2}$", fontsize=10)

    # --- Lambda-omega: uniform except at core ---
    _, _, u_lo, v_lo = make_lo_spiral(N=N, L=L)
    R_lo = np.sqrt(u_lo**2 + v_lo**2)

    im1 = axes[1].imshow(R_lo, extent=[0, L, 0, L], cmap="viridis",
                         origin="lower")
    axes[1].set_title("$\\lambda$-$\\omega$\n$R \\approx \\sqrt{a}$ except at core",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_aspect("equal")
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046)
    cb1.set_label("$R$", fontsize=10)
    axes[1].plot(10, 10, "w*", markersize=8)
    axes[1].annotate("core\n$R \\to 0$", xy=(10, 10), xytext=(13, 14),
                     fontsize=9, color="white", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    # --- FHN: structured amplitude ---
    _, _, u_fhn, v_fhn = make_fhn_spiral(N=N, L=L)
    R_fhn = np.sqrt(u_fhn**2 + v_fhn**2)

    im2 = axes[2].imshow(R_fhn, extent=[0, L, 0, L], cmap="viridis",
                         origin="lower")
    axes[2].set_title("FHN\n$R$ varies across domain", fontsize=12,
                      fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_aspect("equal")
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046)
    cb2.set_label("$R$", fontsize=10)

    fig.suptitle("Amplitude Field $R = \\sqrt{u^2 + v^2}$ Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_08_amplitude_comparison.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  sw_08 done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating spiral wave explainer figures...")
    print(f"Output directory: {OUT_DIR}")
    print()

    fig01_spiral_concept()
    fig02_excitable_vs_oscillatory()
    fig03_fhn_phase_portrait()
    fig04_fhn_spiral()
    fig05_lambda_omega_polar()
    fig06_lambda_omega_spiral()
    fig07_u_vs_laplacian()
    fig08_amplitude_comparison()

    print()
    print(f"Done. {len(list(OUT_DIR.glob('sw_*.png')))} figures generated.")


if __name__ == "__main__":
    main()
