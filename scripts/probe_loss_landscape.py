"""Loss-landscape probe across (MAML, Baseline) × (noise=0, noise=0.01).

For a chosen experiment, picks the top-K tasks where MAML loses to Baseline at
noise=0.01 (largest gap in summed per-mixer holdout NMSE). For each chosen
task, runs fine_tune from {θ*, θ₀} × {noise=0, noise=0.01} and saves the four
adapted state_dicts. Then constructs a 2D trajectory-aligned slice of the loss
surface (Option B: direction-1 = MAML(n0)→MAML(n01), direction-2 =
Gram-Schmidt component of MAML(n0)→Baseline(n01)) and evaluates the
sum-across-mixers raw NMSE on a regular grid at noise=0 and noise=0.01.

Outputs one figure per task (two contour panels side-by-side) into
`<exp_dir>/figures/loss_landscape/`.

Usage:
    python scripts/probe_loss_landscape.py <exp_dir> [--top-k 5] [--grid 41] \\
        [--target-noise 0.01] [--reference-noise 0.0] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Allow importing helpers from scripts/evaluate.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # type: ignore[import]
    load_mixer_from_checkpoint,
    fine_tune,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ExperimentConfig
from src.training.task_loader import TASK_REGISTRY, MetaLearningDataLoader, PDETask
from src.training.imaml import compute_raw_losses


# ============================================================================
# Phase 1 — Task selection
# ============================================================================


def _summed_mse_holdout(
    npz: np.lib.npyio.NpzFile, combo: str, method: str
) -> Optional[float]:
    """Sum the final-step `mse_main_holdout` across mixers for a combo/method.
    Returns None if any per-mixer entry is missing or non-numeric.
    """
    prefix = f"{combo}/{method}/fine_tune/"
    suffix = "/mse_main_holdout"
    keys = [k for k in npz.keys() if k.startswith(prefix) and k.endswith(suffix)]
    if not keys:
        return None
    total = 0.0
    for k in keys:
        v = npz[k]
        if v.dtype.kind != "f" or np.any(np.isnan(v)):
            return None
        total += float(v[-1])
    return total


def select_top_tasks(
    exp_dir: Path, target_noise: float, top_k: int
) -> List[Tuple[str, float, float, float]]:
    """Return [(task_name, gap, maml_total, bl_total)] sorted by gap descending.
    Gap = maml_total − bl_total; positive gap = baseline beat MAML."""
    samples = exp_dir / "evaluation" / "samples"
    rows: List[Tuple[str, float, float, float]] = []

    noise_str = f"{target_noise:.2f}"
    for npz_path in sorted(samples.glob("*.npz")):
        npz = np.load(npz_path, allow_pickle=True)
        combos = sorted(
            {
                k.split("/")[0]
                for k in npz.keys()
                if f"noise_{noise_str}" in k.split("/")[0]
                and not k.split("/")[0].startswith("best_combo")
            }
        )
        if not combos:
            continue
        combo = combos[0]
        m_total = _summed_mse_holdout(npz, combo, "maml")
        b_total = _summed_mse_holdout(npz, combo, "baseline")
        if m_total is None or b_total is None:
            continue
        rows.append((npz_path.stem, m_total - b_total, m_total, b_total))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


# ============================================================================
# Phase 2 — Eval setup + adapted state_dicts
# ============================================================================


def setup_eval(exp_dir: Path, device: str):
    """Load config, theta*, theta0, build cost_function. Returns a dict bundle."""
    cfg = ExperimentConfig.from_yaml(exp_dir / "training" / "config.yaml")

    pde_type = cfg.experiment.pde_type
    task_class = TASK_REGISTRY[pde_type]
    test_dir = Path(cfg.data.meta_test_dir)
    input_mode = cfg.training.input_mode
    test_loader = MetaLearningDataLoader(
        test_dir, task_class=task_class, task_pattern="*_fourier.npz",
        device=device, input_mode=input_mode,
    )
    sizing_task = test_loader.tasks[0]

    theta_star_path = exp_dir / "checkpoints" / "final_model.pt"
    theta_0_path = exp_dir / "checkpoints" / "initial_model.pt"

    aux_losses_enabled = cfg.training.aux_losses_enabled
    hidden_dims = cfg.training.hidden_dims or [350, 350]
    activation = cfg.training.activation or "silu"
    input_bypass = cfg.training.input_bypass

    theta_star = load_mixer_from_checkpoint(
        theta_star_path, device=device, task=sizing_task,
        aux_losses_enabled=aux_losses_enabled,
        hidden_dims=hidden_dims, activation=activation, input_bypass=input_bypass,
    )
    theta_0 = load_mixer_from_checkpoint(
        theta_0_path, device=device, task=sizing_task,
        aux_losses_enabled=aux_losses_enabled,
        hidden_dims=hidden_dims, activation=activation, input_bypass=input_bypass,
    )

    # Loss function (replicates evaluate.main)
    loss_type = cfg.training.loss_function

    def _pw(p, t):
        if loss_type == "mse":
            return F.mse_loss(p, t)
        if loss_type == "normalized_mse":
            return F.mse_loss(p, t) / (t ** 2).mean()
        raise ValueError(f"Unsupported loss_function for probe: {loss_type}")

    def cost_function(pred, target, _coords):
        return _pw(pred, target)

    # Proximal lambda per mixer
    n_outputs = sizing_task.n_outputs
    proximal_lam_list: List[float] = [0.0] * n_outputs
    inner_optimizer = "sgd"
    if cfg.training.imaml.enabled:
        ck = torch.load(theta_star_path, map_location=device, weights_only=False)
        saved_lam = ck.get("lam")
        if saved_lam is not None:
            if isinstance(saved_lam, list):
                proximal_lam_list = [float(t.item()) for t in saved_lam]
            else:
                proximal_lam_list = [float(saved_lam.item())] * n_outputs
        else:
            proximal_lam_list = [float(cfg.training.imaml.lam)] * n_outputs
        inner_optimizer = cfg.training.imaml.inner_optimizer

    anil_mode = "all"
    if cfg.training.imaml.enabled and cfg.training.imaml.anil:
        anil_mode = "head"

    return {
        "cfg": cfg,
        "test_loader": test_loader,
        "sizing_task": sizing_task,
        "theta_star": theta_star,
        "theta_0": theta_0,
        "cost_function": cost_function,
        "aux_losses_enabled": aux_losses_enabled,
        "proximal_lam_list": proximal_lam_list,
        "inner_optimizer": inner_optimizer,
        "anil_mode": anil_mode,
        "n_outputs": n_outputs,
        "device": device,
    }


def adapt_one(
    init_model,
    init_label: str,
    task: PDETask,
    noise: float,
    bundle,
    k_shot: int,
    holdout_size: int,
    max_steps: int,
    fine_tune_lr: float,
    fixed_steps: List[int],
    seed: int,
    noise_idx: int = 0,
):
    """Run fine_tune across all mixers and return the final state_dict.

    `seed` is the per-task k_seed from evaluate.py's scheme (= experiment.seed +
    task_idx * 10000 + k_idx * 100). `noise_idx` is the position of `noise`
    in the eval config's `noise_levels` list — used to reproduce evaluate.py's
    per-noise generator seed (k_seed + noise_idx * 1000).

    init_label ∈ {"maml", "baseline"} controls ANIL (head vs all) and proximal.
    """
    fast_model = copy.deepcopy(init_model)
    n_outputs = bundle["n_outputs"]
    device = bundle["device"]

    noise_gen: Optional[torch.Generator] = None
    if noise > 0.0:
        noise_gen = torch.Generator(device=device).manual_seed(seed + noise_idx * 1000)

    support, holdout, support_coords, holdout_coords = task.get_support_query_split(
        K_shot=k_shot, query_size=holdout_size,
        k_seed=seed, snapshot_seed=seed,
        noise_level=noise, noise_generator=noise_gen,
    )
    support_features_list, support_targets = support
    holdout_features_list, holdout_targets = holdout

    # Snapshot pre-adapt theta for proximal (MAML branch only)
    theta_snapshots = [
        torch.cat([
            p.detach().flatten()
            for p in fast_model.mixer_outer_params(i)  # type: ignore[attr-defined]
        ])
        for i in range(n_outputs)
    ]

    use_anil_for_inner = bundle["anil_mode"] if init_label == "maml" else "all"
    use_lam_per_mixer = bundle["proximal_lam_list"] if init_label == "maml" else [0.0] * n_outputs

    for mixer_idx in range(n_outputs):
        mixer_name = task.mixer_names[mixer_idx]
        fine_tune(
            fast_model=fast_model,
            task=task,
            mixer_idx=mixer_idx,
            mixer_name=mixer_name,
            mixer_outer_params=fast_model.mixer_outer_params(mixer_idx),  # type: ignore[attr-defined]
            inner_params=fast_model.mixer_inner_params(mixer_idx, use_anil_for_inner),  # type: ignore[attr-defined]
            features=support_features_list[mixer_idx],
            targets=support_targets,
            holdout_features=holdout_features_list[mixer_idx],
            holdout_targets=holdout_targets,
            support_coords=support_coords,
            holdout_coords=holdout_coords,
            cost_function=bundle["cost_function"],
            aux_losses_enabled=bundle["aux_losses_enabled"],
            inner_steps=max_steps,
            inner_lr=fine_tune_lr,
            inner_optimizer=bundle["inner_optimizer"],
            lam=use_lam_per_mixer[mixer_idx] if init_label == "maml" else 0.0,
            theta_snapshot=theta_snapshots[mixer_idx] if init_label == "maml" else None,
            fixed_steps=fixed_steps,
            max_grad_norm=0.0,
            slope_recovery_inner=0.0,
        )

    final_state = OrderedDict(
        (k, v.detach().clone()) for k, v in sorted(fast_model.state_dict().items())
    )
    return (
        final_state,
        holdout_features_list,
        holdout_targets,
        support_coords,
        holdout_coords,
    )


# ============================================================================
# Phase 3 — Plane construction + grid evaluation
# ============================================================================


# Keys whose substrings flag "phantom" parameters that should be excluded from the
# trajectory-aligned plane. Kendall log-variances are registered as parameters but
# are frozen at fine_tune time and don't affect raw NMSE — including them in the
# direction vectors dilutes the geometric scale (measured: 63% of θ*↔θ₀ L2
# displacement on LO library is in log_var subspace).
EXCLUDED_KEY_SUBSTRINGS = ("log_variances", "log_var", "log_sigma")


def _is_excluded(key: str) -> bool:
    return any(sub in key for sub in EXCLUDED_KEY_SUBSTRINGS)


def _key_in_scope(
    k: str,
    mixer_filter: Optional[int],
    keep_keys: Optional[set] = None,
) -> bool:
    """True if `k` should participate in the directions.

    Excludes log_vars universally. If `keep_keys` is given, that's the *only*
    inclusion test (mixer_filter is ignored — `keep_keys` is already scoped).
    Otherwise, if `mixer_filter` is set, restricts to that mixer's params.
    """
    if _is_excluded(k):
        return False
    if keep_keys is not None:
        return k in keep_keys
    if mixer_filter is not None and not k.startswith(f"mixers.{mixer_filter}."):
        return False
    return True


def _head_keys_for_mixer(model, mixer_idx: int) -> set:
    """Resolve which state_dict key names correspond to the head subspace of
    a given mixer, by id-matching the parameters returned by
    `model.mixer_inner_params(mixer_idx, "head")` against `named_parameters()`.
    """
    head_params = list(model.mixer_inner_params(mixer_idx, "head"))  # type: ignore[attr-defined]
    head_ids = {id(p) for p in head_params}
    keys: set = set()
    for name, p in model.named_parameters():
        if id(p) in head_ids:
            keys.add(name)
    return keys


def flatten_params(
    state_dict: "OrderedDict[str, torch.Tensor]",
    mixer_filter: Optional[int] = None,
    keep_keys: Optional[set] = None,
) -> torch.Tensor:
    """Concat in-scope tensors into a flat vector. `keep_keys` (if provided) is the
    explicit allow-list of state_dict keys; takes precedence over `mixer_filter`."""
    parts = []
    for k in sorted(state_dict.keys()):
        if not _key_in_scope(k, mixer_filter, keep_keys):
            continue
        parts.append(state_dict[k].detach().to(torch.float64).reshape(-1))
    return torch.cat(parts)


def unflatten_params(
    flat: torch.Tensor,
    reference: "OrderedDict[str, torch.Tensor]",
    mixer_filter: Optional[int] = None,
    keep_keys: Optional[set] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Inverse of flatten_params. Out-of-scope keys are copied verbatim from
    `reference`; in-scope keys are filled from `flat`.
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    cursor = 0
    for k in sorted(reference.keys()):
        ref = reference[k]
        if not _key_in_scope(k, mixer_filter, keep_keys):
            out[k] = ref.detach().clone()
            continue
        n = ref.numel()
        chunk = flat[cursor:cursor + n].reshape(ref.shape).to(dtype=ref.dtype, device=ref.device)
        out[k] = chunk
        cursor += n
    if cursor != flat.numel():
        raise ValueError(
            f"flat has {flat.numel()} elements but reference state_dict has {cursor} in-scope."
        )
    return out


def _project_onto(point_flat: torch.Tensor, origin: torch.Tensor,
                  dir1: torch.Tensor, dir2: torch.Tensor) -> Tuple[float, float]:
    """Project an arbitrary point onto the (dir1, dir2) plane relative to origin."""
    v = point_flat - origin
    return float((v @ dir1).item()), float((v @ dir2).item())


def build_plane(
    sd_maml_n0: "OrderedDict[str, torch.Tensor]",
    sd_maml_n01: "OrderedDict[str, torch.Tensor]",
    sd_bl_n01: "OrderedDict[str, torch.Tensor]",
    sd_theta_star: Optional["OrderedDict[str, torch.Tensor]"] = None,
    sd_theta_zero: Optional["OrderedDict[str, torch.Tensor]"] = None,
    sd_bl_n0: Optional["OrderedDict[str, torch.Tensor]"] = None,
    mixer_filter: Optional[int] = None,
):
    """Option B: trajectory-aligned 2D plane spanning the three landing points.

    Returns (origin_flat, dir1, dir2, coords). `coords` contains the three landing
    points by construction; if `sd_theta_star` / `sd_theta_zero` are provided,
    their orthogonal projections onto (dir1, dir2) are also returned (in general
    they will not lie in the plane — coords give the in-plane projection).
    """
    origin = flatten_params(sd_maml_n0,  mixer_filter=mixer_filter)
    v_maml = flatten_params(sd_maml_n01, mixer_filter=mixer_filter) - origin
    v_bl   = flatten_params(sd_bl_n01,   mixer_filter=mixer_filter) - origin

    n_maml = float(v_maml.norm().item())
    if n_maml < 1e-12:
        raise RuntimeError("MAML(n0) and MAML(n01) coincide — cannot build trajectory plane.")
    dir1 = v_maml / n_maml

    proj1_bl = float((v_bl @ dir1).item())
    perp = v_bl - proj1_bl * dir1
    n_perp = float(perp.norm().item())
    if n_perp < 1e-9 * float(v_bl.norm().item() + 1e-30):
        # Degenerate — three points collinear; fall back to a random orthogonal direction.
        rand = torch.randn_like(dir1)
        rand = rand - (rand @ dir1) * dir1
        n_perp = float(rand.norm().item())
        dir2 = rand / n_perp
        proj2_bl = 0.0
    else:
        dir2 = perp / n_perp
        proj2_bl = n_perp

    coords: Dict[str, Tuple[float, float]] = {
        "maml_n0": (0.0, 0.0),
        "maml_n01": (n_maml, 0.0),
        "bl_n01": (proj1_bl, proj2_bl),
    }

    if sd_theta_star is not None:
        coords["theta_star"] = _project_onto(
            flatten_params(sd_theta_star, mixer_filter=mixer_filter), origin, dir1, dir2,
        )
    if sd_theta_zero is not None:
        coords["theta_zero"] = _project_onto(
            flatten_params(sd_theta_zero, mixer_filter=mixer_filter), origin, dir1, dir2,
        )
    if sd_bl_n0 is not None:
        coords["bl_n0"] = _project_onto(
            flatten_params(sd_bl_n0, mixer_filter=mixer_filter), origin, dir1, dir2,
        )

    return origin, dir1, dir2, coords


def build_plane_pca(
    state_dicts: "OrderedDict[str, OrderedDict[str, torch.Tensor]]",
    mixer_filter: Optional[int] = None,
    keep_keys: Optional[set] = None,
):
    """PCA-basis 2D plane through all landing points.

    Stack flattened (mixer-filtered, log_vars-excluded) parameter vectors of every
    state_dict in `state_dicts` into X ∈ R^(N, P). Origin = column mean. Basis =
    top-2 right singular vectors of (X − origin). Returns
    (origin, dir1, dir2, coords, residuals, explained_var_frac):

      coords[name]    = (α, β) projection
      residuals[name] = ||v − (α·dir1 + β·dir2)||₂   (off-plane norm)
      explained_var_frac = (σ₁² + σ₂²) / Σσᵢ²

    Unlike the trajectory plane, **no point sits exactly in the plane** — every
    landing has some residual. The residuals tell you how literal the in-plane
    marker positions are.
    """
    names = list(state_dicts.keys())
    flats = [
        flatten_params(state_dicts[n], mixer_filter=mixer_filter, keep_keys=keep_keys)
        for n in names
    ]
    X = torch.stack(flats, dim=0)  # (N, P) float64 already
    origin = X.mean(dim=0)
    Xc = X - origin
    # economy SVD: when N < P, Vh is (N, P) — rows are right singular vectors.
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    if Vh.shape[0] < 2:
        raise RuntimeError(f"PCA needs ≥2 singular components; got {Vh.shape[0]}.")
    dir1 = Vh[0]
    dir2 = Vh[1]
    # Renormalize defensively (SVD already orthonormal in principle).
    dir1 = dir1 / (dir1.norm() + 1e-30)
    dir2 = dir2 - (dir2 @ dir1) * dir1
    dir2 = dir2 / (dir2.norm() + 1e-30)

    coords: Dict[str, Tuple[float, float]] = {}
    residuals: Dict[str, float] = {}
    for n, x in zip(names, flats):
        v = x - origin
        a = float((v @ dir1).item())
        b = float((v @ dir2).item())
        coords[n] = (a, b)
        recon = a * dir1 + b * dir2
        residuals[n] = float((v - recon).norm().item())

    total_var = float((S ** 2).sum().item())
    top2_var = float((S[:2] ** 2).sum().item())
    explained = top2_var / max(total_var, 1e-30)
    return origin, dir1, dir2, coords, residuals, explained


def grid_axes(
    coords: Dict[str, Tuple[float, float]], n_grid: int, pad_frac: float = 0.4
):
    """Build (alphas, betas) covering all three points with `pad_frac` margin."""
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    xmin -= pad_frac * xspan
    xmax += pad_frac * xspan
    ymin -= pad_frac * yspan
    ymax += pad_frac * yspan
    return (
        np.linspace(xmin, xmax, n_grid),
        np.linspace(ymin, ymax, n_grid),
    )


@torch.no_grad()
def eval_loss_at(
    model,
    state_dict_template: "OrderedDict[str, torch.Tensor]",
    flat_point: torch.Tensor,
    holdout_features_list: List[torch.Tensor],
    holdout_targets: torch.Tensor,
    n_outputs: int,
    target_mixer: Optional[int] = None,
    mixer_filter: Optional[int] = None,
    keep_keys: Optional[set] = None,
) -> float:
    """Load `flat_point` into `model` and return mixer-target_mixer's NMSE on holdout
    (or summed-mixer NMSE if `target_mixer is None`).

    `mixer_filter` controls which keys the flat vector populates — typically equal
    to `target_mixer` for per-mixer landscapes. Other-mixer parameters are copied
    verbatim from `state_dict_template`. Note: `forward_one(m, …)` only routes
    through mixer m, so mixer-m's NMSE depends only on mixer-m's parameters —
    other-mixer values can be anything without changing the result.
    """
    sd = unflatten_params(flat_point, state_dict_template, mixer_filter=mixer_filter, keep_keys=keep_keys)
    model.load_state_dict(sd)
    model.eval()

    if target_mixer is not None:
        m = target_mixer
        pred = model.forward_one(m, holdout_features_list[m])  # type: ignore[attr-defined]
        target_m = holdout_targets[:, m]
        denom = float((target_m ** 2).mean().item())
        if denom < 1e-30:
            return 0.0
        mse = float(((pred - target_m) ** 2).mean().item())
        return mse / denom

    total = 0.0
    for m in range(n_outputs):
        pred = model.forward_one(m, holdout_features_list[m])  # type: ignore[attr-defined]
        target_m = holdout_targets[:, m]
        denom = float((target_m ** 2).mean().item())
        if denom < 1e-30:
            continue
        mse = float(((pred - target_m) ** 2).mean().item())
        total += mse / denom
    return total


def evaluate_grid(
    model,
    template: "OrderedDict[str, torch.Tensor]",
    origin: torch.Tensor,
    dir1: torch.Tensor,
    dir2: torch.Tensor,
    alphas: np.ndarray,
    betas: np.ndarray,
    holdout_features_list: List[torch.Tensor],
    holdout_targets: torch.Tensor,
    n_outputs: int,
    target_mixer: Optional[int] = None,
    mixer_filter: Optional[int] = None,
    keep_keys: Optional[set] = None,
) -> np.ndarray:
    """Return a (len(betas), len(alphas)) array of NMSE values.

    If `target_mixer` is set, returns mixer-target_mixer's NMSE; otherwise summed
    across mixers. `mixer_filter` controls which subspace the directions live in.
    """
    grid = np.full((len(betas), len(alphas)), np.nan, dtype=np.float64)
    for j, b in enumerate(betas):
        for i, a in enumerate(alphas):
            point = origin + float(a) * dir1 + float(b) * dir2
            try:
                grid[j, i] = eval_loss_at(
                    model, template, point,
                    holdout_features_list, holdout_targets, n_outputs,
                    target_mixer=target_mixer, mixer_filter=mixer_filter,
                    keep_keys=keep_keys,
                )
            except Exception:
                grid[j, i] = np.nan
    return grid


# ============================================================================
# Phase 4 — Plotting
# ============================================================================


def _interp_loss_at(alphas, betas, grid, a, b) -> float:
    """Bilinear-interpolate `grid` at coordinate (a, b). Returns NaN if out of range."""
    if a < alphas[0] or a > alphas[-1] or b < betas[0] or b > betas[-1]:
        return float("nan")
    i = np.searchsorted(alphas, a) - 1
    j = np.searchsorted(betas, b) - 1
    i = max(0, min(i, len(alphas) - 2))
    j = max(0, min(j, len(betas) - 2))
    fa = (a - alphas[i]) / (alphas[i + 1] - alphas[i] + 1e-30)
    fb = (b - betas[j]) / (betas[j + 1] - betas[j] + 1e-30)
    g00, g10 = grid[j, i], grid[j, i + 1]
    g01, g11 = grid[j + 1, i], grid[j + 1, i + 1]
    return float(
        (1 - fa) * (1 - fb) * g00 + fa * (1 - fb) * g10
        + (1 - fa) * fb * g01 + fa * fb * g11
    )


def _plot_panel(ax, alphas, betas, grid, coords, title, panel_noise: str = "n0",
                landing_losses: Optional[Dict[str, float]] = None,
                target_noise: float = 0.01,
                landing_losses_true: Optional[Dict[str, float]] = None):
    """Single panel: log-scale filled contour + line contour overlay + landing/start points.

    `panel_noise` controls which fine_tune trajectory the arrows target — the arrow
    from θ*/θ₀ points at the landing produced at *this* noise level. So the noise=0
    panel shows θ* → MAML(n=0), and the noise=0.01 panel shows θ* → MAML(n=0.01).
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation

    A, B = np.meshgrid(alphas, betas)
    finite = grid[np.isfinite(grid) & (grid > 0)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    vmin = max(np.percentile(finite, 1), 1e-12)
    vmax = np.percentile(finite, 99)
    if not (vmax > vmin):
        vmax = vmin * 10.0
    safe_grid = np.clip(grid, vmin, vmax)

    # Decade-aligned log fill, dense enough to look smooth on a 41-grid.
    lo_dec = np.floor(np.log10(vmin))
    hi_dec = np.ceil(np.log10(vmax))
    n_decades = max(int(hi_dec - lo_dec), 1)
    fill_levels = np.logspace(lo_dec, hi_dec, 12 * n_decades + 1)

    cs = ax.contourf(
        A, B, safe_grid, levels=fill_levels,
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis",
    )
    cb = plt.colorbar(
        cs, ax=ax, fraction=0.046, pad=0.04, label="summed mixer NMSE (log)",
        ticks=LogLocator(base=10, numticks=n_decades + 1),
        format=LogFormatterSciNotation(base=10, labelOnlyBase=True),
    )
    cb.ax.tick_params(labelsize=8)

    # One line contour per decade — labelled — gives crisp readability over the fill.
    line_levels = np.logspace(lo_dec, hi_dec, n_decades + 1)
    line_levels = line_levels[(line_levels >= vmin) & (line_levels <= vmax)]
    if line_levels.size > 0:
        cl = ax.contour(
            A, B, safe_grid, levels=line_levels,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            colors="black", linewidths=0.5, alpha=0.5,
        )
        ax.clabel(cl, fmt=lambda v: f"{v:.0e}", fontsize=6, inline=True)

    # Landing points — filled markers (in-plane by construction).
    # Loss labels: prefer EXACT evaluation passed via `landing_losses` (not bilinear
    # interp from the grid). The grid samples discretely; if the loss-surface basin
    # is narrower than the grid spacing, interp from neighbours overestimates by
    # orders of magnitude. Exact landing losses come from a separate eval_loss_at
    # call at the precise (α, β) of each landing.
    def _label_loss(key: str, a: float, b: float) -> float:
        if landing_losses is not None and key in landing_losses:
            return landing_losses[key]
        return _interp_loss_at(alphas, betas, grid, a, b)

    tgt_label = f"{target_noise:.2f}".rstrip("0").rstrip(".") or "0"
    points = [
        ("theta_star", "#b56cf3",         "θ* (MAML start)"),
        ("theta_zero", "#7ad1f0",         "θ₀ (BL start)"),
        ("maml_n0",    "tab:purple",      "MAML(n=0)"),
        ("maml_n01",   "tab:orange",     f"MAML(n={tgt_label})"),
        ("bl_n0",      "tab:cyan",        "BL(n=0)"),
        ("bl_n01",     "tab:red",        f"BL(n={tgt_label})"),
    ]
    for key, color, label in points:
        if key not in coords:
            continue
        a, b = coords[key]
        loss_here = _label_loss(key, a, b)
        legend_text = f"{label}  L={loss_here:.4e}"
        if landing_losses_true is not None and key in landing_losses_true:
            legend_text += f"  (true: {landing_losses_true[key]:.4e})"
        ax.scatter([a], [b], s=30, marker="o", color=color,
                   edgecolor="black", linewidth=0.4, zorder=7,
                   label=legend_text)

    # Arrows: starts → landings (visualise fine_tune trajectory direction in this plane).
    # Targets are panel-noise-dependent: the noise=0.01 panel shows arrows ending at
    # the noise=0.01 landings, the noise=0 panel ends at noise=0 landings.
    arrow_pairs = [
        ("theta_star", f"maml_{panel_noise}", "tab:purple"),
        ("theta_zero", f"bl_{panel_noise}",   "tab:cyan"),
    ]
    for src_key, dst_key, color in arrow_pairs:
        if src_key in coords and dst_key in coords:
            a0, b0 = coords[src_key]
            a1, b1 = coords[dst_key]
            ax.annotate("", xy=(a1, b1), xytext=(a0, b0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.6))

    ax.set_xlabel(f"α  (dir-1: MAML(n=0) → MAML(n={tgt_label}))", fontsize=9)
    ax.set_ylabel(f"β  (dir-2: ⊥ toward BL(n={tgt_label}))",       fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=7, framealpha=0.9, borderpad=0.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)


def plot_task(
    task_name: str,
    alphas, betas,
    grid_n0, grid_n01,
    coords, gap, m_total, b_total,
    out_path: Path,
    reference_noise: float,
    target_noise: float,
    landing_losses_ref: Optional[Dict[str, float]] = None,
    landing_losses_tgt: Optional[Dict[str, float]] = None,
    landing_losses_true_ref: Optional[Dict[str, float]] = None,
    landing_losses_true_tgt: Optional[Dict[str, float]] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    suptitle = (
        f"{task_name}  —  noise={target_noise:.2f}  Δ={gap:+.4e}   "
        f"(MAML={m_total:.4e}, BL={b_total:.4e})\n"
        "Contour = mixer NMSE on the body+head subspace (log_vars excluded). "
        "In-plane losses are exact at the marker; off-plane points (θ*, θ₀, BL(n=0)) also report true 5000-D loss."
    )
    fig.suptitle(suptitle, fontsize=9)
    _plot_panel(axes[0], alphas, betas, grid_n0,  coords,
                f"Loss surface @ noise={reference_noise:.2f}", panel_noise="n0",
                landing_losses=landing_losses_ref, target_noise=target_noise,
                landing_losses_true=landing_losses_true_ref)
    _plot_panel(axes[1], alphas, betas, grid_n01, coords,
                f"Loss surface @ noise={target_noise:.2f}",    panel_noise="n01",
                landing_losses=landing_losses_tgt, target_noise=target_noise,
                landing_losses_true=landing_losses_true_tgt)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def _pca_marker_style(name: str) -> Tuple[str, str]:
    """(color, display label) for a landing key on a PCA panel.

    Shared 5-color palette across all panels: the branch (MAML vs BL) is
    indicated by the row label, not by color. Color encodes only the role
    within a branch — start point vs noise level.
    """
    # Shared palette — high-contrast hues, all distinct.
    PALETTE = {
        "start":    "#000000",  # black
        "n_0.00":   "#1f77b4",  # blue
        "n_0.01":   "#ff7f0e",  # orange
        "n_0.05":   "#2ca02c",  # green
        "n_0.10":   "#d62728",  # red
    }
    if name == "theta_star":
        return (PALETTE["start"], "θ* (start)")
    if name == "theta_zero":
        return (PALETTE["start"], "θ₀ (start)")
    # Format: "<init>_n<noise:.2f>"  e.g. "maml_n0.01"
    if "_n" in name:
        init, ns = name.split("_n")
        try:
            noise = float(ns)
        except ValueError:
            return ("#888888", name)
        key = f"n_{noise:.2f}"
        color = PALETTE.get(key, "#888888")
        prefix = "MAML" if init == "maml" else ("BL" if init == "baseline" else init)
        return (color, f"{prefix}(n={noise:.2f})")
    return ("#888888", name)


def _plot_panel_pca(ax, alphas, betas, grid, coords, residuals, explained,
                    title, eval_noise: float,
                    landing_losses: Optional[Dict[str, float]] = None,
                    landing_losses_true: Optional[Dict[str, float]] = None):
    """Single PCA-basis panel: log filled+line contour + every landing point."""
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation

    A, B = np.meshgrid(alphas, betas)
    finite = grid[np.isfinite(grid) & (grid > 0)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    vmin = max(np.percentile(finite, 1), 1e-12)
    vmax = np.percentile(finite, 99)
    if not (vmax > vmin):
        vmax = vmin * 10.0
    safe_grid = np.clip(grid, vmin, vmax)

    lo_dec = np.floor(np.log10(vmin))
    hi_dec = np.ceil(np.log10(vmax))
    n_decades = max(int(hi_dec - lo_dec), 1)
    fill_levels = np.logspace(lo_dec, hi_dec, 12 * n_decades + 1)

    cs = ax.contourf(
        A, B, safe_grid, levels=fill_levels,
        norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis",
    )
    cb = plt.colorbar(
        cs, ax=ax, fraction=0.046, pad=0.04, label="mixer NMSE (log)",
        ticks=LogLocator(base=10, numticks=n_decades + 1),
        format=LogFormatterSciNotation(base=10, labelOnlyBase=True),
    )
    cb.ax.tick_params(labelsize=8)

    line_levels = np.logspace(lo_dec, hi_dec, n_decades + 1)
    line_levels = line_levels[(line_levels >= vmin) & (line_levels <= vmax)]
    if line_levels.size > 0:
        cl = ax.contour(
            A, B, safe_grid, levels=line_levels,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            colors="black", linewidths=0.5, alpha=0.5,
        )
        ax.clabel(cl, fmt=lambda v: f"{v:.0e}", fontsize=6, inline=True)

    # Every landing in coords gets a marker.
    # Sort: starts first, then maml-by-noise, then bl-by-noise (legend readability)
    def _sort_key(name: str) -> Tuple[int, float]:
        if name == "theta_star":
            return (0, 0.0)
        if name == "theta_zero":
            return (0, 1.0)
        if name.startswith("maml_n"):
            return (1, float(name.split("_n")[1]))
        if name.startswith("baseline_n"):
            return (2, float(name.split("_n")[1]))
        return (3, 0.0)

    for name in sorted(coords.keys(), key=_sort_key):
        a, b = coords[name]
        color, lab = _pca_marker_style(name)
        in_plane = landing_losses.get(name, np.nan) if landing_losses else np.nan
        true_loss = landing_losses_true.get(name, np.nan) if landing_losses_true else np.nan
        resid = residuals.get(name, np.nan)
        # Compact legend entry.
        legend_text = f"{lab}  Lᵢₚ={in_plane:.2e}  Lₜ={true_loss:.2e}  r={resid:.2e}"
        ax.scatter([a], [b], s=32, marker="o", color=color,
                   edgecolor="black", linewidth=0.4, zorder=7,
                   label=legend_text)

    ax.set_xlabel("α  (PC1)", fontsize=9)
    ax.set_ylabel("β  (PC2)", fontsize=9)
    ax.set_title(f"{title}\nexplained var (PC1+PC2) = {explained * 100:.1f}%", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=6.5, framealpha=0.9, borderpad=0.4)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)


def plot_task_pca_split(
    task_name: str,
    maml_pd: Dict, bl_pd: Dict,
    gap: float, m_total: float, b_total: float,
    out_path: Path,
    reference_noise: float,
    target_noise: float,
):
    """4-panel figure: row 0 = MAML branch (ref / target), row 1 = BL branch."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    suptitle = (
        f"{task_name}  —  PCA (split)  —  noise={target_noise:.2f}  Δ={gap:+.4e}   "
        f"(MAML={m_total:.4e}, BL={b_total:.4e})\n"
        "Top row: MAML branch (basis from 4 adapted MAML heads, θ* as overlay).   "
        "Bottom row: BL branch (basis from 4 adapted BL body+head, θ₀ as overlay).\n"
        "Lᵢₚ = loss at projected (α,β);  Lₜ = loss at true 5000-D point;  r = ||off-plane residual||."
    )
    fig.suptitle(suptitle, fontsize=9)

    for row, pd in enumerate([maml_pd, bl_pd]):
        branch = "MAML" if row == 0 else "BL"
        for col, (grid, ll, ll_true, n) in enumerate([
            (pd["grid_ref"], pd["landing_losses_ref"], pd["landing_losses_true_ref"], reference_noise),
            (pd["grid_tgt"], pd["landing_losses_tgt"], pd["landing_losses_true_tgt"], target_noise),
        ]):
            _plot_panel_pca(
                axes[row, col],
                pd["alphas"], pd["betas"], grid,
                pd["coords"], pd["residuals"], pd["explained"],
                f"{branch} branch — Loss surface @ noise={n:.2f}",
                eval_noise=n,
                landing_losses=ll,
                landing_losses_true=ll_true,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def _compute_pca_pass(
    *,
    basis_label: str,
    origin: torch.Tensor, dir1: torch.Tensor, dir2: torch.Tensor,
    coords: Dict[str, Tuple[float, float]],
    residuals: Dict[str, float],
    explained: float,
    frozen_body_template: "OrderedDict[str, torch.Tensor]",
    keep_keys: Optional[set],
    true_sources: Dict[str, "OrderedDict[str, torch.Tensor]"],
    eval_model,
    args,
    target_noise: float,
    hf_ref, ht_ref, hf_tgt, ht_tgt,
    n_outputs: int,
    mixer_idx: int,
) -> Dict:
    """Compute grid + exact landing losses + true losses for one PCA branch.

    `frozen_body_template` is the state_dict whose out-of-`keep_keys` parameters
    are held fixed for every grid evaluation. For head-only PCA (MAML branch,
    `keep_keys = head_keys`) this should be θ* — the body stays frozen at the
    meta-trained checkpoint, so the (α, β) sweep moves only the head. For
    full-parameter PCA (BL branch, `keep_keys = None`) every parameter is in
    scope and the template body is unused but still required by `unflatten_params`.

    Returns a pass-data dict consumed by `plot_task_pca_split` and `_save_pca_pass_npz`.
    """
    template = frozen_body_template
    mixer_arg = mixer_idx if keep_keys is None else None
    alphas, betas = grid_axes(coords, args.grid, pad_frac=args.pad_frac)
    print(f"      [{basis_label}] grid {args.grid}×{args.grid}  "
          f"α∈[{alphas[0]:.3e},{alphas[-1]:.3e}]  "
          f"β∈[{betas[0]:.3e},{betas[-1]:.3e}]", flush=True)
    print(f"      [{basis_label}] evaluating reference noise={args.reference_noise:.2f} ...",
          flush=True)
    grid_ref = evaluate_grid(
        eval_model, template, origin, dir1, dir2, alphas, betas,
        hf_ref, ht_ref, n_outputs,
        target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
    )
    print(f"      [{basis_label}] evaluating target noise={target_noise:.2f} ...", flush=True)
    grid_tgt = evaluate_grid(
        eval_model, template, origin, dir1, dir2, alphas, betas,
        hf_tgt, ht_tgt, n_outputs,
        target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
    )

    def _exact_loss(point_coords, hf_holdout, ht_holdout):
        a, b = point_coords
        flat_pt = origin + float(a) * dir1 + float(b) * dir2
        return eval_loss_at(
            eval_model, template, flat_pt,
            hf_holdout, ht_holdout, n_outputs,
            target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
        )

    landing_losses_ref = {n: _exact_loss(coords[n], hf_ref, ht_ref) for n in coords}
    landing_losses_tgt = {n: _exact_loss(coords[n], hf_tgt, ht_tgt) for n in coords}

    def _true_loss(sd, hf_holdout, ht_holdout):
        flat_pt = flatten_params(sd, mixer_filter=mixer_arg, keep_keys=keep_keys)
        return eval_loss_at(
            eval_model, template, flat_pt,
            hf_holdout, ht_holdout, n_outputs,
            target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
        )

    landing_losses_true_ref = {n: _true_loss(sd, hf_ref, ht_ref)
                               for n, sd in true_sources.items()}
    landing_losses_true_tgt = {n: _true_loss(sd, hf_tgt, ht_tgt)
                               for n, sd in true_sources.items()}

    return {
        "basis_label": basis_label,
        "alphas": alphas, "betas": betas,
        "grid_ref": grid_ref, "grid_tgt": grid_tgt,
        "coords": coords, "residuals": residuals, "explained": explained,
        "landing_losses_ref": landing_losses_ref,
        "landing_losses_tgt": landing_losses_tgt,
        "landing_losses_true_ref": landing_losses_true_ref,
        "landing_losses_true_tgt": landing_losses_true_tgt,
    }


def _save_pca_pass_npz(pd: Dict, out_path: Path, mixer_idx: int, args, target_noise: float):
    """Write per-branch npz for a PCA pass."""
    coords = pd["coords"]
    residuals = pd["residuals"]
    landing_losses_ref = pd["landing_losses_ref"]
    landing_losses_tgt = pd["landing_losses_tgt"]
    landing_losses_true_ref = pd["landing_losses_true_ref"]
    landing_losses_true_tgt = pd["landing_losses_true_tgt"]
    coord_keys = list(coords.keys())
    true_keys  = list(landing_losses_true_ref.keys())
    np.savez(
        out_path,
        alphas=pd["alphas"], betas=pd["betas"],
        grid_reference=pd["grid_ref"], grid_target=pd["grid_tgt"],
        coord_keys=np.array(coord_keys),
        coords=np.array([coords[k] for k in coord_keys]),
        residuals=np.array([residuals.get(k, 0.0) for k in coord_keys]),
        landing_losses_ref=np.array([landing_losses_ref.get(k, np.nan) for k in coord_keys]),
        landing_losses_tgt=np.array([landing_losses_tgt.get(k, np.nan) for k in coord_keys]),
        landing_loss_true_keys=np.array(true_keys),
        landing_losses_true_ref=np.array([landing_losses_true_ref.get(k, np.nan) for k in true_keys]),
        landing_losses_true_tgt=np.array([landing_losses_true_tgt.get(k, np.nan) for k in true_keys]),
        explained_var_top2=float(pd["explained"]),
        basis=pd["basis_label"],
        mixer_idx=mixer_idx,
        reference_noise=args.reference_noise, target_noise=target_noise,
    )


def _run_landscape_pass(
    *,
    basis_label: str,
    origin: torch.Tensor, dir1: torch.Tensor, dir2: torch.Tensor,
    coords: Dict[str, Tuple[float, float]],
    residuals: Dict[str, float],
    explained: float,
    template: "OrderedDict[str, torch.Tensor]",
    keep_keys: Optional[set],
    true_sources: Dict[str, "OrderedDict[str, torch.Tensor]"],
    eval_model,
    args,
    hf_ref, ht_ref, hf_tgt, ht_tgt,
    n_outputs: int,
    mixer_idx: int,
    task_name: str,
    mixer_name: str,
    t_key: str,
    target_noise: float,
    out_dir: Path,
    gap_caption: Tuple[float, float, float],
):
    """Run grid eval + exact/true loss + plot + npz for the trajectory basis.

    `basis_label` is currently always "trajectory" — PCA branches go through
    `_compute_pca_pass` + `plot_task_pca_split` instead. The label is still used
    as the figure-tag in console output and the npz `basis` field. `residuals`
    and `explained` are accepted for signature symmetry with the PCA path; for
    trajectory they are written verbatim into the npz (residuals={} → zeros,
    explained=1.0 by caller convention, since the three trajectory anchors are
    in-plane by construction).
    """
    t_gap, t_m_total, t_b_total = gap_caption
    mixer_arg = mixer_idx if keep_keys is None else None

    alphas, betas = grid_axes(coords, args.grid, pad_frac=args.pad_frac)
    print(f"      [{basis_label}] grid {args.grid}×{args.grid}  "
          f"α∈[{alphas[0]:.3e},{alphas[-1]:.3e}]  "
          f"β∈[{betas[0]:.3e},{betas[-1]:.3e}]", flush=True)
    print(f"      [{basis_label}] evaluating reference noise={args.reference_noise:.2f} ...",
          flush=True)
    grid_ref = evaluate_grid(
        eval_model, template, origin, dir1, dir2, alphas, betas,
        hf_ref, ht_ref, n_outputs,
        target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
    )
    print(f"      [{basis_label}] evaluating target noise={target_noise:.2f} ...",
          flush=True)
    grid_tgt = evaluate_grid(
        eval_model, template, origin, dir1, dir2, alphas, betas,
        hf_tgt, ht_tgt, n_outputs,
        target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
    )

    def _exact_loss(point_coords, hf_holdout, ht_holdout):
        a, b = point_coords
        flat_pt = origin + float(a) * dir1 + float(b) * dir2
        return eval_loss_at(
            eval_model, template, flat_pt,
            hf_holdout, ht_holdout, n_outputs,
            target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
        )

    landing_losses_ref = {n: _exact_loss(coords[n], hf_ref, ht_ref) for n in coords}
    landing_losses_tgt = {n: _exact_loss(coords[n], hf_tgt, ht_tgt) for n in coords}

    def _true_loss(sd, hf_holdout, ht_holdout):
        flat_pt = flatten_params(sd, mixer_filter=mixer_arg, keep_keys=keep_keys)
        return eval_loss_at(
            eval_model, template, flat_pt,
            hf_holdout, ht_holdout, n_outputs,
            target_mixer=mixer_idx, mixer_filter=mixer_arg, keep_keys=keep_keys,
        )

    landing_losses_true_ref = {n: _true_loss(sd, hf_ref, ht_ref)
                               for n, sd in true_sources.items()}
    landing_losses_true_tgt = {n: _true_loss(sd, hf_tgt, ht_tgt)
                               for n, sd in true_sources.items()}

    fname = f"{task_name}_mixer{mixer_idx}_{mixer_name}_target{t_key}_landscape"
    out_path = out_dir / f"{fname}.png"

    plot_task(
        f"{task_name} — mixer {mixer_idx} ({mixer_name})",
        alphas, betas, grid_ref, grid_tgt, coords,
        t_gap, t_m_total, t_b_total, out_path,
        args.reference_noise, target_noise,
        landing_losses_ref=landing_losses_ref,
        landing_losses_tgt=landing_losses_tgt,
        landing_losses_true_ref=landing_losses_true_ref,
        landing_losses_true_tgt=landing_losses_true_tgt,
    )

    coord_keys = list(coords.keys())
    true_keys  = list(landing_losses_true_ref.keys())
    np.savez(
        out_dir / f"{fname}.npz",
        alphas=alphas, betas=betas,
        grid_reference=grid_ref, grid_target=grid_tgt,
        coord_keys=np.array(coord_keys),
        coords=np.array([coords[k] for k in coord_keys]),
        residuals=np.array([residuals.get(k, 0.0) for k in coord_keys]),
        landing_losses_ref=np.array([landing_losses_ref.get(k, np.nan) for k in coord_keys]),
        landing_losses_tgt=np.array([landing_losses_tgt.get(k, np.nan) for k in coord_keys]),
        landing_loss_true_keys=np.array(true_keys),
        landing_losses_true_ref=np.array([landing_losses_true_ref.get(k, np.nan) for k in true_keys]),
        landing_losses_true_tgt=np.array([landing_losses_true_tgt.get(k, np.nan) for k in true_keys]),
        explained_var_top2=float(explained),
        basis=basis_label,
        mixer_idx=mixer_idx,
        reference_noise=args.reference_noise, target_noise=target_noise,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_dir", type=str, help="Path to experiment directory")
    p.add_argument("--top-k", type=int, default=5, help="Number of tasks to probe")
    p.add_argument("--grid", type=int, default=41, help="Grid resolution per axis")
    p.add_argument("--pad-frac", type=float, default=0.25,
                   help="Fractional padding around landing points")
    p.add_argument("--target-noise", type=float, nargs="+", default=[0.01, 0.05, 0.10],
                   help="Target noise levels (one figure per target, right panel). "
                        "Each must appear in the eval config's noise_levels.")
    p.add_argument("--reference-noise", type=float, default=0.0,
                   help="Reference noise level (left panel)")
    p.add_argument("--selection-noise", type=float, default=0.01,
                   help="Noise level used for top-K task selection (gap = MAML-BL at this noise)")
    p.add_argument("--basis", type=str, default="trajectory",
                   choices=["trajectory", "pca"],
                   help="Plane construction. 'trajectory' = Gram-Schmidt through "
                        "{MAML(n=0), MAML(n=tgt), BL(n=tgt)}. 'pca' = top-2 SVD "
                        "of all 10 landings centered at their mean.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--k-shot", type=int, default=None,
                   help="Override k_shot for probe (default: from config)")
    p.add_argument("--holdout-size", type=int, default=None,
                   help="Override holdout_size for probe (default: from config)")
    args = p.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    out_dir = exp_dir / "figures" / "loss_landscape"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1 — task selection ============================================
    print(f"\n[1/4] Selecting top-{args.top_k} tasks at noise={args.selection_noise:.2f}")
    chosen = select_top_tasks(exp_dir, args.selection_noise, args.top_k)
    if not chosen:
        print("  No tasks found. Exiting.")
        return
    for name, gap, m, b in chosen:
        print(f"      {name}  Δ={gap:+.4e}   MAML={m:.4e}  BL={b:.4e}")

    # === Phase 2 — eval setup ================================================
    print(f"\n[2/4] Loading experiment setup on device={args.device}")
    bundle = setup_eval(exp_dir, args.device)
    cfg = bundle["cfg"]
    ev = cfg.evaluation
    k_shot = args.k_shot if args.k_shot is not None else (ev.k_values[0] if ev.k_values else 4000)
    holdout_size = args.holdout_size if args.holdout_size is not None else ev.holdout_size
    max_steps = ev.max_steps
    fine_tune_lr = ev.fine_tune_lr
    # L-BFGS requires fixed_steps == [0, max_steps] only (per fine_tune validation)
    fixed_steps = [0, max_steps] if bundle["inner_optimizer"] == "lbfgs" else [0, max_steps]
    print(f"      k_shot={k_shot}  holdout={holdout_size}  steps={max_steps}  lr={fine_tune_lr}")
    print(f"      inner_optimizer={bundle['inner_optimizer']}  anil_mode={bundle['anil_mode']}")

    # Index tasks by name and remember each task's positional task_idx so we can
    # reproduce evaluate.py's seeding scheme exactly: seed = experiment.seed + task_idx * 10000
    name_to_task: Dict[str, PDETask] = {
        t.task_name: t for t in bundle["test_loader"].tasks
    }
    name_to_idx: Dict[str, int] = {
        t.task_name: i for i, t in enumerate(bundle["test_loader"].tasks)
    }
    base_seed = int(cfg.experiment.seed)
    print(f"      base seed (experiment.seed): {base_seed}")

    # === Phase 3 — adapt + landscape per chosen task =========================
    print(f"\n[3/4] Per-task adaptation + landscape evaluation")
    for ti, (task_name, gap, m_total, b_total) in enumerate(chosen):
        print(f"\n  ({ti + 1}/{len(chosen)}) {task_name}")
        if task_name not in name_to_task:
            print(f"    SKIP — task not in test_loader (looking for '{task_name}')")
            continue
        task = name_to_task[task_name]

        # Reproduce evaluate.py seeding for THIS task. There's only one k_value (k_idx=0)
        # so k_seed == per_task_seed; that's what we pass into adapt_one.
        per_task_seed = base_seed + name_to_idx[task_name] * 10000
        print(f"    per-task seed (eval-matching): {per_task_seed}")

        # Map each chosen noise to its position in the eval config's noise_levels
        # so we reproduce evaluate.py's `k_seed + noise_idx * 1000` generator seed.
        eval_noise_levels = list(cfg.evaluation.noise_levels)

        def _noise_idx_of(n: float) -> int:
            for i, nl in enumerate(eval_noise_levels):
                if abs(nl - n) < 1e-6:
                    return i
            raise ValueError(
                f"noise={n} not in eval config noise_levels={eval_noise_levels}; "
                "match the probe's reference/target noise to one of those values."
            )

        # Run all needed fine_tunes once; reused across target-noise comparisons.
        all_noises = [args.reference_noise] + list(args.target_noise)
        # De-duplicate while preserving order (in case reference is in target list)
        seen: set = set()
        unique_noises = [n for n in all_noises if not (n in seen or seen.add(n))]

        sds: Dict[str, "OrderedDict[str, torch.Tensor]"] = {}
        holdouts: Dict[str, Tuple] = {}
        for label, init_model in [("maml", bundle["theta_star"]), ("baseline", bundle["theta_0"])]:
            for noise in unique_noises:
                key = f"{label}_n{noise:.2f}"
                noise_idx = _noise_idx_of(noise)
                print(f"    fine_tune  init={label}  noise={noise:.2f}  (eval noise_idx={noise_idx})", flush=True)
                sd, hf, ht, _, _ = adapt_one(
                    init_model, label, task, noise, bundle,
                    k_shot=k_shot, holdout_size=holdout_size,
                    max_steps=max_steps, fine_tune_lr=fine_tune_lr,
                    fixed_steps=fixed_steps, seed=per_task_seed,
                    noise_idx=noise_idx,
                )
                sds[key] = sd
                holdouts[key] = (hf, ht)
                if args.device == "cuda":
                    torch.cuda.empty_cache()

        # Pull the un-adapted starting points (frozen meta-trained checkpoints) as
        # OrderedDict state_dicts in the same key order we use elsewhere.
        sd_theta_star = OrderedDict(
            (k, v.detach().clone()) for k, v in sorted(bundle["theta_star"].state_dict().items())
        )
        sd_theta_zero = OrderedDict(
            (k, v.detach().clone()) for k, v in sorted(bundle["theta_0"].state_dict().items())
        )

        # Use the same theta_star model object as a stateful evaluator (load_state_dict each call)
        eval_model = copy.deepcopy(bundle["theta_star"])
        template = sds[f"maml_n{args.reference_noise:.2f}"]
        hf_ref, ht_ref = holdouts[f"maml_n{args.reference_noise:.2f}"]

        # Outer loop: one figure per (target_noise, mixer). Each target_noise gets its
        # own trajectory plane built from the landings at that specific noise level.
        for target_noise in args.target_noise:
            t_key = f"{target_noise:.2f}"
            print(f"  ── target noise={target_noise:.2f} ──")
            hf_tgt, ht_tgt = holdouts[f"maml_n{target_noise:.2f}"]

            # Per-target gap (for the figure caption)
            t_gap, t_m_total, t_b_total = gap, m_total, b_total
            samples = exp_dir / "evaluation" / "samples"
            npz_path = samples / f"{task_name}.npz"
            if npz_path.exists():
                npz = np.load(npz_path, allow_pickle=True)
                noise_str = f"{target_noise:.2f}"
                combos = sorted(
                    {k.split("/")[0] for k in npz.keys()
                     if f"noise_{noise_str}" in k.split("/")[0]
                     and not k.split("/")[0].startswith("best_combo")}
                )
                if combos:
                    m_t = _summed_mse_holdout(npz, combos[0], "maml")
                    b_t = _summed_mse_holdout(npz, combos[0], "baseline")
                    if m_t is not None and b_t is not None:
                        t_m_total, t_b_total = m_t, b_t
                        t_gap = m_t - b_t

            # Per-mixer probe at this target noise
            for mixer_idx in range(bundle["n_outputs"]):
                mixer_name = name_to_task[task_name].mixer_names[mixer_idx]
                print(f"    --- mixer {mixer_idx} ({mixer_name}) @ noise={target_noise:.2f} "
                      f"[basis={args.basis}] ---", flush=True)

                # State_dicts available for this (task, mixer):
                #   sd_theta_star, sd_theta_zero, sds[maml|baseline_n*]
                # PCA mode splits into two branches with different scopes:
                #   warm: head-only flatten, points = {θ*, MAML(n_i)}    (body frozen under ANIL)
                #   cold: full body+head, points = {θ₀, BL(n_i)}         (BL adapts everything)
                # Each branch builds its own basis, plots its own contour figure.

                if args.basis == "trajectory":
                    try:
                        origin, dir1, dir2, coords = build_plane(
                            sds[f"maml_n{args.reference_noise:.2f}"],
                            sds[f"maml_n{target_noise:.2f}"],
                            sds[f"baseline_n{target_noise:.2f}"],
                            sd_theta_star=sd_theta_star,
                            sd_theta_zero=sd_theta_zero,
                            sd_bl_n0=sds[f"baseline_n{args.reference_noise:.2f}"],
                            mixer_filter=mixer_idx,
                        )
                    except RuntimeError as e:
                        print(f"      SKIP — {e}")
                        continue
                    n_dir1 = coords["maml_n01"][0]
                    n_dir2 = coords["bl_n01"][1]
                    print(f"      ||dir-1|| (head shift n=0→n={target_noise:.2f}) = {n_dir1:.4e}")
                    print(f"      ||dir-2|| (⊥ shift toward BL)              = {n_dir2:.4e}")

                    _run_landscape_pass(
                        basis_label="trajectory",
                        origin=origin, dir1=dir1, dir2=dir2,
                        coords=coords, residuals={}, explained=1.0,
                        template=template, keep_keys=None,
                        true_sources={
                            "theta_star": sd_theta_star,
                            "theta_zero": sd_theta_zero,
                            "bl_n0":      sds[f"baseline_n{args.reference_noise:.2f}"],
                        },
                        eval_model=eval_model, args=args,
                        hf_ref=hf_ref, ht_ref=ht_ref, hf_tgt=hf_tgt, ht_tgt=ht_tgt,
                        n_outputs=bundle["n_outputs"], mixer_idx=mixer_idx,
                        task_name=task_name, mixer_name=mixer_name, t_key=t_key,
                        target_noise=target_noise, out_dir=out_dir,
                        gap_caption=(t_gap, t_m_total, t_b_total),
                    )
                    continue

                # ----- PCA mode: MAML + BL split, merged into one figure -----
                # Each branch's basis fits from its 4 adapted points; the start
                # point (θ* on MAML, θ₀ on BL) is projected onto the resulting
                # plane as an off-plane overlay marker.
                head_keys = _head_keys_for_mixer(eval_model, mixer_idx)

                maml_basis_sds: "OrderedDict[str, OrderedDict[str, torch.Tensor]]" = OrderedDict()
                for nz in unique_noises:
                    maml_basis_sds[f"maml_n{nz:.2f}"] = sds[f"maml_n{nz:.2f}"]

                bl_basis_sds: "OrderedDict[str, OrderedDict[str, torch.Tensor]]" = OrderedDict()
                for nz in unique_noises:
                    bl_basis_sds[f"baseline_n{nz:.2f}"] = sds[f"baseline_n{nz:.2f}"]

                pca_passes = [
                    # (branch_label, scope_label, basis_sds, keep_keys, template,
                    #  overlay_name, overlay_sd)
                    ("MAML", "head-only", maml_basis_sds, head_keys, sd_theta_star,
                     "theta_star", sd_theta_star),
                    ("BL",   "body+head", bl_basis_sds,   None,      sd_theta_zero,
                     "theta_zero", sd_theta_zero),
                ]

                pass_data: Dict[str, Dict] = {}
                for (branch_label, scope_label, basis_sds, branch_keep, branch_template,
                     overlay_name, overlay_sd) in pca_passes:
                    mixer_arg_b = mixer_idx if branch_keep is None else None
                    print(f"      [{branch_label}] PCA scope={scope_label}, "
                          f"{len(basis_sds)} basis points + 1 overlay ({overlay_name})", flush=True)
                    try:
                        origin_b, dir1_b, dir2_b, coords_b, residuals_b, explained_b = build_plane_pca(
                            basis_sds,
                            mixer_filter=mixer_arg_b,
                            keep_keys=branch_keep,
                        )
                    except RuntimeError as e:
                        print(f"      SKIP {branch_label} — {e}")
                        continue

                    # Project overlay (start checkpoint) onto the basis plane.
                    overlay_flat = flatten_params(
                        overlay_sd, mixer_filter=mixer_arg_b, keep_keys=branch_keep,
                    )
                    v = overlay_flat - origin_b
                    a_o = float((v @ dir1_b).item())
                    b_o = float((v @ dir2_b).item())
                    recon = a_o * dir1_b + b_o * dir2_b
                    coords_b[overlay_name] = (a_o, b_o)
                    residuals_b[overlay_name] = float((v - recon).norm().item())

                    print(f"      [{branch_label}] explained var (PC1+PC2) = {explained_b * 100:.1f}%, "
                          f"max basis residual = {max(residuals_b[k] for k in basis_sds):.4e}, "
                          f"{overlay_name} overlay residual = {residuals_b[overlay_name]:.4e}")

                    true_sources = dict(basis_sds)
                    true_sources[overlay_name] = overlay_sd

                    pd = _compute_pca_pass(
                        basis_label=f"pca_{branch_label}",
                        origin=origin_b, dir1=dir1_b, dir2=dir2_b,
                        coords=coords_b, residuals=residuals_b, explained=explained_b,
                        frozen_body_template=branch_template, keep_keys=branch_keep,
                        true_sources=true_sources,
                        eval_model=eval_model, args=args,
                        target_noise=target_noise,
                        hf_ref=hf_ref, ht_ref=ht_ref, hf_tgt=hf_tgt, ht_tgt=ht_tgt,
                        n_outputs=bundle["n_outputs"], mixer_idx=mixer_idx,
                    )
                    pass_data[branch_label] = pd

                if "MAML" in pass_data and "BL" in pass_data:
                    fname_base = f"{task_name}_mixer{mixer_idx}_{mixer_name}_target{t_key}_landscape_pca"
                    out_path = out_dir / f"{fname_base}.png"
                    plot_task_pca_split(
                        f"{task_name} — mixer {mixer_idx} ({mixer_name})",
                        pass_data["MAML"], pass_data["BL"],
                        t_gap, t_m_total, t_b_total, out_path,
                        args.reference_noise, target_noise,
                    )
                    _save_pca_pass_npz(
                        pass_data["MAML"], out_dir / f"{fname_base}_MAML.npz",
                        mixer_idx, args, target_noise,
                    )
                    _save_pca_pass_npz(
                        pass_data["BL"], out_dir / f"{fname_base}_BL.npz",
                        mixer_idx, args, target_noise,
                    )

    print(f"\n[4/4] Done. Figures: {out_dir}")


if __name__ == "__main__":
    main()
