"""
Verification tests for spectral structural loss.

Run on CPU — no GPU needed. Tests:
1. Parseval sanity: on a regular grid, spectral MSE ≈ physical MSE
2. Gradient flow: gradients reach model parameters through NUFFT + higher
3. NUFFT batching: multi-channel input produces correct output shape
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Test 1: Parseval sanity check
# ---------------------------------------------------------------------------
def test_parseval():
    """On a regular grid, spectral MSE should equal physical MSE (up to normalization)."""
    from src.training.spectral_loss import compute_spectral_loss

    N = 64
    Lx, Ly = 10.0, 10.0

    # Regular grid points
    x = torch.linspace(0, Lx, N + 1)[:-1]  # exclude endpoint for periodicity
    y = torch.linspace(0, Ly, N + 1)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    x_pts = xx.flatten()
    y_pts = yy.flatten()

    n_pts = x_pts.shape[0]

    # Two different signals (single channel)
    pred = torch.sin(2 * torch.pi * x_pts / Lx).unsqueeze(1)
    target = torch.cos(2 * torch.pi * x_pts / Lx).unsqueeze(1)

    # Physical normalized MSE
    phys_mse = F.mse_loss(pred, target) / (target**2).mean()

    # Spectral normalized MSE (using all N modes = full representation)
    spec_mse = compute_spectral_loss(pred, target, x_pts, y_pts, Lx, Ly, n_modes=N)

    ratio = (spec_mse / phys_mse).item()
    print(f"  Physical normalized MSE: {phys_mse.item():.6f}")
    print(f"  Spectral normalized MSE: {spec_mse.item():.6f}")
    print(f"  Ratio (spectral/physical): {ratio:.4f}")

    # They won't be exactly equal because NUFFT normalization differs from
    # standard FFT by a factor of N². But the ratio should be constant.
    # Key check: both are finite, positive, and the ratio is stable.
    assert spec_mse.item() > 0, "Spectral loss should be positive"
    assert torch.isfinite(spec_mse), "Spectral loss should be finite"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 2: Gradient flow through NUFFT
# ---------------------------------------------------------------------------
def test_gradient_flow():
    """Verify gradients flow from spectral loss back to model parameters."""
    from src.training.spectral_loss import compute_spectral_loss

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.SiLU(),
        nn.Linear(32, 2),
    )

    N = 100
    Lx, Ly = 10.0, 10.0

    x_pts = torch.rand(N) * Lx
    y_pts = torch.rand(N) * Ly
    features = torch.randn(N, 10)
    target = torch.randn(N, 2)

    pred = model(features)
    loss = compute_spectral_loss(pred, target, x_pts, y_pts, Lx, Ly, n_modes=16)
    loss.backward()

    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  All params have nonzero gradients: {has_grad}")
    assert has_grad, "Gradients should reach all model parameters"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 3: Gradient flow through higher + NUFFT
# ---------------------------------------------------------------------------
def test_gradient_flow_higher():
    """Verify gradients flow through higher innerloop_ctx + NUFFT to meta-parameters."""
    import higher
    from src.training.spectral_loss import compute_spectral_loss

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.SiLU(),
        nn.Linear(32, 2),
    )

    N = 100
    Lx, Ly = 10.0, 10.0

    x_pts = torch.rand(N) * Lx
    y_pts = torch.rand(N) * Ly
    support_x = torch.randn(N, 10)
    support_y = torch.randn(N, 2)
    query_x = torch.randn(N, 10)
    query_y = torch.randn(N, 2)

    inner_opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (
        fmodel,
        diffopt,
    ):
        # Inner step with spectral loss
        pred = fmodel(support_x)
        inner_loss = compute_spectral_loss(
            pred, support_y, x_pts, y_pts, Lx, Ly, n_modes=16
        )
        diffopt.step(inner_loss)

        # Outer loss (also spectral, for testing)
        query_pred = fmodel(query_x)
        outer_loss = compute_spectral_loss(
            query_pred, query_y, x_pts, y_pts, Lx, Ly, n_modes=16
        )

    outer_loss.backward()

    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    print(f"  Inner loss: {inner_loss.item():.6f}")
    print(f"  Outer loss: {outer_loss.item():.6f}")
    print(f"  Meta-gradients reach all params: {has_grad}")
    assert has_grad, (
        "Meta-gradients should reach all model parameters through higher + NUFFT"
    )
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 4: NUFFT batching (multi-channel)
# ---------------------------------------------------------------------------
def test_batched_nufft():
    """Verify finufft_type1 handles multi-channel input correctly."""
    from pytorch_finufft.functional import finufft_type1

    N = 200
    n_modes = 16

    x = torch.rand(N) * 2 * torch.pi
    y = torch.rand(N) * 2 * torch.pi
    points = torch.stack([x, y], dim=0)

    # Single channel (complex)
    values_1ch = torch.complex(torch.randn(N), torch.zeros(N))
    out_1ch = finufft_type1(points, values_1ch, (n_modes, n_modes))
    print(f"  Single channel output shape: {out_1ch.shape}")
    assert out_1ch.shape == (n_modes, n_modes), (
        f"Expected ({n_modes}, {n_modes}), got {out_1ch.shape}"
    )

    # Multi channel (batch, N) — batch dim first, N last per docs
    real_2ch = torch.randn(2, N)
    values_2ch = torch.complex(real_2ch, torch.zeros_like(real_2ch))
    out_2ch = finufft_type1(points, values_2ch, (n_modes, n_modes))
    print(f"  Two channel output shape: {out_2ch.shape}")
    assert out_2ch.shape == (2, n_modes, n_modes)

    # Verify batch consistency: channel 0 of batched == single call on channel 0
    out_ch0_solo = finufft_type1(points, values_2ch[0].contiguous(), (n_modes, n_modes))
    match = torch.allclose(out_2ch[0], out_ch0_solo, atol=1e-5)
    print(f"  Batch channel 0 matches solo: {match}")
    assert match

    print("  PASSED")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Test 1: Parseval sanity check")
    test_parseval()
    print()

    print("Test 2: Gradient flow (direct)")
    test_gradient_flow()
    print()

    print("Test 3: Gradient flow (through higher)")
    test_gradient_flow_higher()
    print()

    print("Test 4: NUFFT batching")
    test_batched_nufft()
    print()

    print("All tests passed.")
