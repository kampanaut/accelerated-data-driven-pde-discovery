"""
PDE residual sanity check for Navier-Stokes data.

Verifies that stored derivatives are internally consistent with the PDE:
    u_t = -u*u_x - v*u_y + nu*(u_xx + u_yy) - dp/dx
    v_t = -u*v_x - v*v_y + nu*(v_xx + v_yy) - dp/dy

Note: Pressure gradient is not stored, so the residual has an irreducible
floor from the pressure term. The point is to compare NS residual magnitude
against BR to see if derivative quality explains Jacobian recovery difference.

Usage:
    python scripts/residual_check_ns.py data/datasets/ns_train-1/gv_n5_s1_001.npz
"""

import sys
import numpy as np


def load_task(npz_path: str) -> dict:
    """Load a task .npz and extract arrays + parameters."""
    data = np.load(npz_path, allow_pickle=True)

    sim = data['simulation_params'].item()

    return {
        'u': data['u'],
        'v': data['v'],
        'u_x': data['u_x'],
        'u_y': data['u_y'],
        'u_xx': data['u_xx'],
        'u_yy': data['u_yy'],
        'v_x': data['v_x'],
        'v_y': data['v_y'],
        'v_xx': data['v_xx'],
        'v_yy': data['v_yy'],
        'u_t': data['u_t'],
        'v_t': data['v_t'],
        'nu': sim['nu'],
        'resolution': sim['resolution'],
        'domain_size': sim['domain_size'],
    }


def compute_residuals(task: dict) -> dict:
    """Compute PDE residuals for both u and v equations (ignoring pressure)."""
    u, v = task['u'], task['v']
    u_x, u_y = task['u_x'], task['u_y']
    u_xx, u_yy = task['u_xx'], task['u_yy']
    v_x, v_y = task['v_x'], task['v_y']
    v_xx, v_yy = task['v_xx'], task['v_yy']
    u_t, v_t = task['u_t'], task['v_t']
    nu = task['nu']

    # Reconstruct (minus pressure gradient)
    u_advection = -(u * u_x + v * u_y)
    u_diffusion = nu * (u_xx + u_yy)
    u_t_reconstructed = u_advection + u_diffusion

    v_advection = -(u * v_x + v * v_y)
    v_diffusion = nu * (v_xx + v_yy)
    v_t_reconstructed = v_advection + v_diffusion

    u_residual = u_t - u_t_reconstructed  # includes -dp/dx
    v_residual = v_t - v_t_reconstructed  # includes -dp/dy

    return {
        'u_residual_mse': float(np.mean(u_residual**2)),
        'v_residual_mse': float(np.mean(v_residual**2)),
        'u_residual_max': float(np.max(np.abs(u_residual))),
        'v_residual_max': float(np.max(np.abs(v_residual))),
        'u_advection_rms': float(np.sqrt(np.mean(u_advection**2))),
        'u_diffusion_rms': float(np.sqrt(np.mean(u_diffusion**2))),
        'v_advection_rms': float(np.sqrt(np.mean(v_advection**2))),
        'v_diffusion_rms': float(np.sqrt(np.mean(v_diffusion**2))),
        'u_relative_mse': float(np.mean(u_residual**2) / np.mean(u_t**2)),
        'v_relative_mse': float(np.mean(v_residual**2) / np.mean(v_t**2)),
        '_u_residual': u_residual,
        '_v_residual': v_residual,
        '_u_t': u_t,
        '_v_t': v_t,
        '_u_diffusion': u_diffusion,
        '_u_advection': u_advection,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/residual_check_ns.py <path_to_npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    print(f"Loading: {npz_path}")

    task = load_task(npz_path)

    res = task['resolution']
    dom = task['domain_size']
    dx = dom[0] / res[0]
    print(f"Resolution: {res[0]}x{res[1]}, domain: {dom[0]:.4f}x{dom[1]:.4f}")
    print(f"dx = {dx:.4f}, dx² = {dx**2:.6f}")
    print(f"nu = {task['nu']:.6f}")
    print(f"Samples: {len(task['u']):,}")
    print(f"NOTE: Residual includes pressure gradient (not stored)")
    print()

    results = compute_residuals(task)

    print("=== PDE Residual Check (NS) ===")
    for key, val in results.items():
        if key.startswith('_'):
            continue
        if isinstance(val, float):
            print(f"  {key}: {val:.6e}")
        else:
            print(f"  {key}: {val}")

    # Spatial analysis
    nx, ny = task['resolution']
    n_spatial = nx * ny
    u_res = results['_u_residual']
    v_res = results['_v_residual']
    n_timesteps = len(u_res) // n_spatial

    print(f"\n=== Spatial Analysis ({n_timesteps} timesteps x {nx}x{ny}) ===")
    u_res_grid = u_res.reshape(n_timesteps, ny, nx)
    v_res_grid = v_res.reshape(n_timesteps, ny, nx)

    margin = 2
    u_interior = u_res_grid[:, margin:-margin, margin:-margin]
    v_interior = v_res_grid[:, margin:-margin, margin:-margin]
    u_boundary = np.mean(u_res_grid**2) - np.mean(u_interior**2)

    print(f"  u interior MSE: {np.mean(u_interior**2):.6e}")
    print(f"  v interior MSE: {np.mean(v_interior**2):.6e}")
    print(f"  u boundary excess: {u_boundary:.6e}")

    # Temporal
    mid = n_timesteps // 2
    print(f"\n=== Temporal Analysis ===")
    print(f"  u early MSE (t<mid): {np.mean(u_res_grid[:mid]**2):.6e}")
    print(f"  u late MSE  (t>mid): {np.mean(u_res_grid[mid:]**2):.6e}")
    print(f"  v early MSE (t<mid): {np.mean(v_res_grid[:mid]**2):.6e}")
    print(f"  v late MSE  (t>mid): {np.mean(v_res_grid[mid:]**2):.6e}")

    # Error source correlation
    u_diff = results['_u_diffusion']
    u_adv = results['_u_advection']
    corr_diff = np.corrcoef(u_res.flatten(), u_diff.flatten())[0, 1]
    corr_adv = np.corrcoef(u_res.flatten(), u_adv.flatten())[0, 1]
    print(f"\n=== Error Source ===")
    print(f"  u_residual corr with diffusion term:  {corr_diff:.4f}")
    print(f"  u_residual corr with advection term:  {corr_adv:.4f}")
    print(f"  (residual ~= pressure gradient for NS)")


if __name__ == '__main__':
    main()
