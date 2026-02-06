"""
PDE residual sanity check for Brusselator data.

Verifies that stored derivatives are internally consistent with the PDE:
    u_t = D_A * (u_xx + u_yy) + k1 - (k2+1)*u + u^2*v
    v_t = D_B * (v_xx + v_yy) + k2*u - u^2*v

If residual is small, the finite-difference derivatives stored in the .npz
match the temporal evolution, and a network *can* learn the correct coefficients.

Usage:
    python scripts/residual_check.py data/datasets/br_train-1/br_gp_001.npz
"""

import sys
import numpy as np


def load_task(npz_path: str) -> dict:
    """Load a task .npz and extract arrays + parameters."""
    data = np.load(npz_path, allow_pickle=True)

    sim = data['simulation_params'].item()
    return {
        # Fields and derivatives (flattened: n_snapshots * nx * ny)
        'u': data['u'],
        'v': data['v'],
        'u_xx': data['u_xx'],
        'u_yy': data['u_yy'],
        'v_xx': data['v_xx'],
        'v_yy': data['v_yy'],
        'u_t': data['u_t'],
        'v_t': data['v_t'],
        # PDE parameters
        'D_A': sim['D_A'],
        'D_B': sim['D_B'],
        'k1': sim['k1'],
        'k2': sim['k2'],
        # Grid info
        'resolution': sim['resolution'],
        'domain_size': sim['domain_size'],
    }


def compute_residuals(task: dict) -> dict:
    """Compute PDE residuals for both u and v equations."""
    u, v = task['u'], task['v']
    u_xx, u_yy = task['u_xx'], task['u_yy']
    v_xx, v_yy = task['v_xx'], task['v_yy']
    u_t, v_t = task['u_t'], task['v_t']
    D_A, D_B = task['D_A'], task['D_B']
    k1, k2 = task['k1'], task['k2']

    # Reconstruct u_t and v_t from stored spatial derivatives + known params
    u_diffusion = D_A * (u_xx + u_yy)
    u_reaction = k1 - (k2 + 1) * u + u**2 * v
    u_t_reconstructed = u_diffusion + u_reaction

    v_diffusion = D_B * (v_xx + v_yy)
    v_reaction = k2 * u - u**2 * v
    v_t_reconstructed = v_diffusion + v_reaction

    u_residual = u_t - u_t_reconstructed
    v_residual = v_t - v_t_reconstructed

    return {
        # Overall residuals
        'u_residual_mse': float(np.mean(u_residual**2)),
        'v_residual_mse': float(np.mean(v_residual**2)),
        'u_residual_max': float(np.max(np.abs(u_residual))),
        'v_residual_max': float(np.max(np.abs(v_residual))),
        # Per-term magnitudes (to see what dominates)
        'u_diffusion_rms': float(np.sqrt(np.mean(u_diffusion**2))),
        'u_reaction_rms': float(np.sqrt(np.mean(u_reaction**2))),
        'v_diffusion_rms': float(np.sqrt(np.mean(v_diffusion**2))),
        'v_reaction_rms': float(np.sqrt(np.mean(v_reaction**2))),
        # Relative error: residual / signal
        'u_relative_mse': float(np.mean(u_residual**2) / np.mean(u_t**2)),
        'v_relative_mse': float(np.mean(v_residual**2) / np.mean(v_t**2)),
        # Raw arrays for further analysis
        '_u_residual': u_residual,
        '_v_residual': v_residual,
        '_u_t': u_t,
        '_v_t': v_t,
        '_u_diffusion': u_diffusion,
        '_u_reaction': u_reaction,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/residual_check.py <path_to_npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    print(f"Loading: {npz_path}")

    task = load_task(npz_path)

    res = task['resolution']
    dom = task['domain_size']
    dx = dom[0] / res[0]
    print(f"Resolution: {res[0]}x{res[1]}, domain: {dom[0]}x{dom[1]}")
    print(f"dx = {dx:.4f}, dx² = {dx**2:.4f}")
    print(f"D_A = {task['D_A']:.4f}, D_B = {task['D_B']:.4f}")
    print(f"k1 = {task['k1']:.4f}, k2 = {task['k2']:.4f}")
    print(f"Samples: {len(task['u']):,}")
    print()

    results = compute_residuals(task)

    if results is None:
        print("compute_residuals() returned None — implement it first")
        sys.exit(1)

    print("=== PDE Residual Check ===")
    for key, val in results.items():
        if key.startswith('_'):
            continue
        if isinstance(val, float):
            print(f"  {key}: {val:.6e}")
        else:
            print(f"  {key}: {val}")

    # Spatial analysis: boundary vs interior
    nx, ny = task['resolution']
    n_spatial = nx * ny
    u_res = results['_u_residual']
    v_res = results['_v_residual']
    n_timesteps = len(u_res) // n_spatial

    print(f"\n=== Spatial Analysis ({n_timesteps} timesteps x {nx}x{ny}) ===")
    u_res_grid = u_res.reshape(n_timesteps, ny, nx)
    v_res_grid = v_res.reshape(n_timesteps, ny, nx)

    # Interior (skip 2 cells from each edge — np.gradient boundary effects)
    margin = 2
    u_interior = u_res_grid[:, margin:-margin, margin:-margin]
    v_interior = v_res_grid[:, margin:-margin, margin:-margin]
    u_boundary = np.mean(u_res_grid**2) - np.mean(u_interior**2)

    print(f"  u interior MSE: {np.mean(u_interior**2):.6e}")
    print(f"  v interior MSE: {np.mean(v_interior**2):.6e}")
    print(f"  u boundary excess: {u_boundary:.6e}")

    # Temporal distribution: early vs late
    mid = n_timesteps // 2
    print(f"\n=== Temporal Analysis ===")
    print(f"  u early MSE (t<mid): {np.mean(u_res_grid[:mid]**2):.6e}")
    print(f"  u late MSE  (t>mid): {np.mean(u_res_grid[mid:]**2):.6e}")
    print(f"  v early MSE (t<mid): {np.mean(v_res_grid[:mid]**2):.6e}")
    print(f"  v late MSE  (t>mid): {np.mean(v_res_grid[mid:]**2):.6e}")

    # Check: is it the diffusion or reaction term that's wrong?
    u_diff = results['_u_diffusion']
    u_react = results['_u_reaction']
    u_t_stored = results['_u_t']
    # Correlation of residual with each term
    corr_diff = np.corrcoef(u_res.flatten(), u_diff.flatten())[0, 1]
    corr_react = np.corrcoef(u_res.flatten(), u_react.flatten())[0, 1]
    print(f"\n=== Error Source ===")
    print(f"  u_residual corr with diffusion term: {corr_diff:.4f}")
    print(f"  u_residual corr with reaction term:  {corr_react:.4f}")
    print(f"  (high |corr| => that term is the error source)")


if __name__ == '__main__':
    main()
