#!/usr/bin/env python3
"""
Evaluation script for MAML vs baseline comparison.

This script:
1. Loads meta-learned θ* and initial θ₀ from checkpoints
2. Loads meta-test tasks
3. For each (task, K, noise) combination:
   - Samples K points ONCE (fair comparison)
   - Fine-tunes from θ* (MAML) and θ₀ (baseline)
   - Records loss trajectories, coefficient estimates, prediction errors
4. Saves results.json (metadata) and samples/*.npz (arrays)

Usage:
    python scripts/evaluate.py --config configs/experiment.yaml
"""

import io
import sys
import copy
import json
import shutil
import argparse


from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Callable, Optional

import torch
torch.set_num_threads(9)
torch.set_num_interop_threads(4)
import torch.nn.functional as F
import numpy as np

from src.config import ExperimentConfig
from src.networks.pde_operator_network import MixerNetwork
from src.training.task_loader import (
    MetaLearningDataLoader,
    PDETask,
    TASK_REGISTRY,
    CoefficientExtraction,
)
from src.training.maml import LSLRSchedule
from src.training.imaml import kendall_total_loss
from src.evaluation.metrics import compress_step_ranges
from src.evaluation.eval_types import MixerFineTuneOutput, assemble_method_result
from src.evaluation.results import (
    MethodResult,
    ComboResult,
    TaskResult,
    WorseFlags,
)


class _TeeStream:
    """Write to both stdout and a StringIO buffer."""

    def __init__(self, original: object):  # type: ignore[arg-type]
        self.original = original
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.original.write(text)  # type: ignore[union-attr]
        self.buffer.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()  # type: ignore[union-attr]

    def getvalue(self) -> str:
        return self.buffer.getvalue()



def load_mixer_from_checkpoint(
    checkpoint_path: Path,
    device: str,
    task: PDETask,
    aux_losses_enabled: bool,
    hidden_dims: List[int],
    activation: str,
    input_bypass: bool,
) -> MixerNetwork:
    """Load a MixerNetwork from a checkpoint, matching the trainer's construction path.

    Both θ* (`final_model.pt`) and θ₀ (`initial_model.pt`) are loaded through
    this function. The model is built via `MixerNetwork.from_task(task, ...)`
    using the same factory the trainer calls in `scripts/train_maml.py`, so
    θ₀ and θ* have identical structure — same number of mixers, same
    per-mixer input_dim derived from the task's `structural_feature_names`,
    same log-variance registration (present iff `aux_losses_enabled`).

    When the checkpoint carries a `config` field (saved by
    `iMAMLTrainer.save_checkpoint`, present in `final_model.pt` but not in
    `initial_model.pt`), enforce that the checkpoint's
    `training.aux_losses_enabled` matches the current one. Loading an
    aux-on checkpoint into aux-off eval would silently drop Kendall
    log-variances via `load_state_dict`'s missing-keys handling; the hard
    check surfaces the mismatch as a `RuntimeError` instead.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        # The trainer saves self.config which is a TrainingSection (not a
        # full ExperimentConfig), so aux_losses_enabled is a direct attribute.
        ckpt_cfg = checkpoint["config"]
        ckpt_aux = ckpt_cfg.aux_losses_enabled
        if ckpt_aux != aux_losses_enabled:
            raise RuntimeError(
                f"aux_losses_enabled mismatch at {checkpoint_path}: "
                f"checkpoint was trained with aux_losses_enabled={ckpt_aux}, "
                f"current eval config has {aux_losses_enabled}. Model "
                f"structure differs (Kendall log-variances registered vs "
                f"not) so the checkpoint cannot be loaded into this evaluator."
            )

    model = MixerNetwork.from_task(
        task,
        aux_losses_enabled=aux_losses_enabled,
        hidden_dims=hidden_dims,
        activation=activation,
        input_bypass=input_bypass,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def fine_tune(
    fast_model: MixerNetwork,
    task: PDETask,
    mixer_idx: int,
    mixer_name: str,
    mixer_outer_params: List[torch.nn.Parameter],
    inner_params: List[torch.nn.Parameter],
    features: torch.Tensor,
    targets: torch.Tensor,
    holdout_features: torch.Tensor,
    holdout_targets: torch.Tensor,
    support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    holdout_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    *,
    cost_function: Callable[
        [torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
        torch.Tensor,
    ],
    aux_losses_enabled: bool,
    inner_steps: int,
    inner_lr: float,
    inner_optimizer: str,
    lam: float,
    theta_snapshot: Optional[torch.Tensor],
    fixed_steps: List[int],
    max_grad_norm: float = 0.0,
    slope_recovery_inner: float = 0.0,
) -> MixerFineTuneOutput:
    """Per-mixer inner solve for evaluation, mirroring iMAML training.

    Adapts one mixer of `fast_model` in place by minimizing

        kendall_total_loss(fast_model, mixer_idx, ...) + λ/2 · ||φ - θ||²

    where φ is the flat of `mixer_outer_params` and θ is `theta_snapshot`.
    The proximal term is active when `lam > 0` and `theta_snapshot is not None`
    (θ* branch with a loaded λ from checkpoint). It's inactive when `lam = 0`
    or `theta_snapshot is None` (θ₀ baseline branch — no meta-init to pull
    toward).

    At each `fixed_steps` index, snapshots:
    - Per-coefficient recoveries via `task.extract_coefficients(mixer_idx, ...)`
      evaluated on the HOLDOUT split (generalization-facing).
    - Prediction residuals `pred - target[:, mixer_idx]` on the HOLDOUT split.

    L-BFGS inner optimizer REQUIRES `fixed_steps == [0, inner_steps]` exactly.
    Intermediate snapshots would force chunking and break the quasi-Newton
    history across chunks, producing a different post-adapt point than
    training. Use `inner_optimizer='sgd'` if you want finer snapshots.

    Train/holdout loss arrays have shape `(inner_steps + 1,)` for both inner
    optimizers in SGD mode. For L-BFGS mode the shape is `(2,)` — pre-adapt
    at index 0, post-adapt at index 1 — because L-BFGS runs all inner_steps
    inside one `opt.step(closure)` call with no per-step exposed hooks.
    Downstream consumers should dispatch on `len(array)` vs `inner_steps + 1`
    if they need to line up curves from both modes.

    Returns a `MixerFineTuneOutput` with per-mixer loss trajectories +
    per-step extraction snapshots + per-step prediction residuals.
    """
    fast_model.train()
    device = features.device

    # Validate L-BFGS fixed_steps — reject non-endpoint per the plan doc.
    if inner_optimizer == "lbfgs":
        expected = {0, inner_steps}
        if set(fixed_steps) != expected:
            raise ValueError(
                f"L-BFGS inner optimizer requires fixed_steps == [0, {inner_steps}] "
                f"(got {sorted(fixed_steps)}). Intermediate snapshots would force "
                f"chunking and break quasi-Newton history across chunks, diverging "
                f"from training behavior. Use inner_optimizer='sgd' if you want "
                f"finer-grained per-step snapshots."
            )

    fixed_steps_set = set(fixed_steps)

    # Kendall-weighted total loss closure — single source of truth with the
    # trainer via the module-level `kendall_total_loss` function.
    def _kendall_loss(
        feat: torch.Tensor,
        tgt: torch.Tensor,
        coords_arg: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        return kendall_total_loss(
            fast_model, mixer_idx, feat, tgt, coords_arg,
            cost_function=cost_function,
            aux_losses_enabled=aux_losses_enabled,
            task=task,
        )

    # Proximal term: 0.5 * λ * ||φ - θ||² over this mixer's outer slice.
    def _proximal_loss() -> torch.Tensor:
        if lam > 0 and theta_snapshot is not None:
            phi = torch.cat([p.view(-1) for p in mixer_outer_params])
            return 0.5 * lam * (phi - theta_snapshot).pow(2).sum()
        return torch.tensor(0.0, device=device)

    # N-LAAF slope recovery for this mixer only (no cross-mixer coupling).
    def _slope_recovery_loss() -> torch.Tensor:
        if slope_recovery_inner > 0:
            return slope_recovery_inner * fast_model.mixer_slope_recovery(mixer_idx)  # type: ignore[attr-defined]
        return torch.tensor(0.0, device=device)

    # Snapshot helper — fires at each configured fixed_step.
    per_step_extractions: Dict[int, Dict[str, Dict[str, CoefficientExtraction]]] = {}
    pred_errors_per_step: Dict[int, np.ndarray] = {}

    def _snapshot(step_num: int) -> None:
        if step_num not in fixed_steps_set:
            return
        # Coefficient recovery on the HOLDOUT split — that's the generalization
        # signal we care about for downstream per-step reconciliation.
        per_step_extractions[step_num] = task.extract_coefficients(
            mixer_idx, fast_model, holdout_features
        )
        # Prediction residuals on the holdout for the prediction-scatter viz.
        with torch.no_grad():
            pred = fast_model.forward_one(mixer_idx, holdout_features)  # type: ignore[attr-defined]
            target_i = holdout_targets[:, mixer_idx]
            pred_errors_per_step[step_num] = (pred - target_i).detach().cpu().numpy()

    train_losses: List[float] = []
    holdout_losses: List[float] = []

    # Step 0: record pre-adapt losses + snapshot.
    # No torch.no_grad wrapper — `_kendall_loss` uses `task.auxiliary_losses`
    # internally which builds a create_graph=True autograd path for the JVP
    # targets (fails inside no_grad). We rely on .item() to detach scalars.
    train_loss_0 = _kendall_loss(features, targets, support_coords)
    train_losses.append(train_loss_0.item())
    del train_loss_0

    holdout_loss_0 = _kendall_loss(holdout_features, holdout_targets, holdout_coords)
    holdout_losses.append(holdout_loss_0.item())
    del holdout_loss_0

    _snapshot(0)

    # Inner solve
    if inner_optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            inner_params, lr=1.0,
            max_iter=inner_steps,
            tolerance_grad=1e-15,
            tolerance_change=1e-15,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            task_loss = _kendall_loss(features, targets, support_coords)
            task_loss = task_loss + _slope_recovery_loss()
            total = task_loss + _proximal_loss()
            total.backward()
            return total

        opt.step(closure)  # type: ignore[arg-type]

        # Post-adapt losses
        train_loss_post = _kendall_loss(features, targets, support_coords)
        train_losses.append(train_loss_post.item())
        del train_loss_post

        holdout_loss_post = _kendall_loss(holdout_features, holdout_targets, holdout_coords)
        holdout_losses.append(holdout_loss_post.item())
        del holdout_loss_post

        _snapshot(inner_steps)

    elif inner_optimizer == "sgd":
        opt = torch.optim.SGD(inner_params, lr=inner_lr)
        for step in range(inner_steps):
            opt.zero_grad()
            task_loss = _kendall_loss(features, targets, support_coords)
            task_loss = task_loss + _slope_recovery_loss()
            total = task_loss + _proximal_loss()
            total.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(inner_params, max_grad_norm)
            opt.step()

            # Record post-step loss + snapshot
            step_num = step + 1
            tl = _kendall_loss(features, targets, support_coords)
            train_losses.append(tl.item())
            del tl
            hl = _kendall_loss(holdout_features, holdout_targets, holdout_coords)
            holdout_losses.append(hl.item())
            del hl

            _snapshot(step_num)
    else:
        raise ValueError(
            f"Unknown inner_optimizer: {inner_optimizer!r}. Use 'sgd' or 'lbfgs'."
        )

    return MixerFineTuneOutput(
        mixer_name=mixer_name,
        train_losses=np.array(train_losses, dtype=np.float64),
        holdout_losses=np.array(holdout_losses, dtype=np.float64),
        per_step_extractions=per_step_extractions,
        pred_errors_per_step=pred_errors_per_step,
    )


def _compute_worse_flags(
    maml: MethodResult,
    baseline: MethodResult,
    fixed_steps: List[int],
    inner_steps: int,
) -> WorseFlags:
    """Compare MAML to baseline — flag steps where MAML did worse.

    Two signals:
    - `loss_steps`: steps where MAML's mean holdout loss (averaged across
      mixers) exceeds baseline's. For L-BFGS inner optimizer, the
      `per_mixer_holdout_losses` arrays have length 2 and we index by
      position in `sorted(fixed_steps)`. For SGD, they have length
      `inner_steps + 1` and we index directly by the step value.
    - `coeff_steps[name]`: steps where MAML's `abs_error` for coefficient
      `name` exceeds baseline's. Indexes into each coefficient's `per_step`
      list positionally.
    """
    loss_worse_steps: List[int] = []
    coeff_worse_steps: Dict[str, List[int]] = {}
    sorted_fixed_steps = sorted(fixed_steps)

    def _loss_at_step(arr: np.ndarray, step_pos: int, step_val: int) -> float:
        """Pick the holdout loss at the given step.

        SGD arrays have length inner_steps + 1 — index by step_val.
        L-BFGS arrays have length 2 — index by step_pos (0 → pre, 1 → post).
        """
        if len(arr) == inner_steps + 1:
            return float(arr[step_val])
        # L-BFGS length-2 path
        return float(arr[step_pos])

    # Loss-level comparison — mean across mixers at each step
    for step_pos, step_val in enumerate(sorted_fixed_steps):
        maml_h = float(np.mean([
            _loss_at_step(arr, step_pos, step_val)
            for arr in maml.fine_tune.per_mixer_holdout_losses.values()
        ]))
        baseline_h = float(np.mean([
            _loss_at_step(arr, step_pos, step_val)
            for arr in baseline.fine_tune.per_mixer_holdout_losses.values()
        ]))
        if maml_h > baseline_h:
            loss_worse_steps.append(step_val)

    # Per-coefficient comparison — abs_error positionally in per_step list
    for coeff_name, maml_snap in maml.coefficient_recovery.coefficients.items():
        baseline_snap = baseline.coefficient_recovery.coefficients.get(coeff_name)
        if baseline_snap is None:
            continue
        for step_pos, step_val in enumerate(sorted_fixed_steps):
            if step_pos >= len(maml_snap.per_step) or step_pos >= len(baseline_snap.per_step):
                continue
            if maml_snap.per_step[step_pos].abs_error > baseline_snap.per_step[step_pos].abs_error:
                coeff_worse_steps.setdefault(coeff_name, []).append(step_val)

    return WorseFlags(loss_steps=loss_worse_steps, coeff_steps=coeff_worse_steps)


def evaluate_task(
    task: PDETask,
    theta_star: MixerNetwork,
    theta_0: MixerNetwork,
    k_values: List[int],
    noise_levels: List[float],
    fine_tune_lr: float,
    max_steps: int,
    device: str,
    seed: int,
    fixed_steps: List[int],
    holdout_size: int = 1000,
    *,
    cost_function: Optional[Callable[
        [torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
        torch.Tensor,
    ]] = None,
    aux_losses_enabled: bool = False,
    proximal_lam: float = 0.0,
    inner_optimizer: str = "lbfgs",
    anil_mode: str = "head",
    max_grad_norm: float = 0.0,
    slope_recovery_inner: float = 0.0,
    loss_type: str = "normalized_mse",
    spectral_mode_size: int = 0,
    log_weights: bool = False,
    lslr: Optional["LSLRSchedule"] = None,
) -> TaskResult:
    """
    Evaluate one task across all (K, noise) combinations.

    For each combo:
      1. Sample support + holdout via `task.get_support_query_split` with
         source-level noise injection (M5b hook).
      2. Deep-copy θ* and θ₀ into fresh adaptation targets.
      3. Snapshot per-mixer theta for the θ* branch's proximal term.
      4. Loop over mixers: call `fine_tune` per mixer on each branch,
         collecting `MixerFineTuneOutput` instances.
      5. Merge per-mixer outputs into `MethodResult` via
         `assemble_method_result` — handles cross-mixer reconciliation
         path keying.
      6. Build `ComboResult` with maml + baseline + worse flags.

    Returns a `TaskResult` with all combos, per-combo per-mixer losses,
    per-step recovery snapshots with cross-path reconciliation, and
    per-path raw values in the NPZ side.
    """
    if cost_function is None:
        raise ValueError("evaluate_task requires an explicit cost_function")

    task_result = TaskResult(
        task_name=task.task_name,
        coefficients=task.true_coefficients,
        ic_type=task.ic_config.get("type", "unknown"),
        n_samples=task.n_samples,
    )

    mixer_names = task.mixer_names
    n_outputs = task.n_outputs
    anil_mode_effective = anil_mode  # fine_tune receives the inner-params slice directly

    true_coefficients = task.true_coefficients

    for k_idx, k in enumerate(k_values):
        k_seed = seed + k_idx * 100
        available_for_holdout = task.n_samples - k
        actual_holdout = min(holdout_size, available_for_holdout)

        for noise_idx, noise in enumerate(noise_levels):
            print(f"    K={k:4d}, noise={noise:.0%}...", end=" ", flush=True)

            try:
                # Source-level noise injection via the M5b hook — sampled
                # into u_hat/v_hat BEFORE derivative computation, so
                # structural features inherit correlated noise from the
                # perturbed state. Noise generator is seeded per combo for
                # reproducibility.
                noise_generator: Optional[torch.Generator] = None
                if noise > 0.0:
                    noise_generator = torch.Generator(device=device).manual_seed(
                        k_seed + noise_idx * 1000
                    )

                support, holdout, support_coords, holdout_coords = task.get_support_query_split(
                    K_shot=k,
                    query_size=actual_holdout,
                    k_seed=k_seed,
                    snapshot_seed=seed,
                    noise_level=noise,
                    noise_generator=noise_generator,
                )
                support_features_list, support_targets = support
                holdout_features_list, holdout_targets = holdout

                # Fresh adaptation targets — deepcopy so each combo starts
                # from the meta-trained weights independent of previous combos.
                maml_model = copy.deepcopy(theta_star)
                baseline_model = copy.deepcopy(theta_0)

                # Per-mixer pre-adapt theta snapshots for the MAML branch's
                # proximal term. Taken BEFORE any mixer gets adapted.
                # Baseline branch runs with lam=0 and theta_snapshot=None —
                # no meta-init to pull toward.
                maml_theta_snapshots: List[torch.Tensor] = [
                    torch.cat([
                        p.detach().flatten()
                        for p in maml_model.mixer_outer_params(i)  # type: ignore[attr-defined]
                    ])
                    for i in range(n_outputs)
                ]

                maml_outputs: List[MixerFineTuneOutput] = []
                baseline_outputs: List[MixerFineTuneOutput] = []

                for mixer_idx in range(n_outputs):
                    mixer_name = mixer_names[mixer_idx]
                    support_x_i = support_features_list[mixer_idx]
                    holdout_x_i = holdout_features_list[mixer_idx]

                    # θ* branch — proximal term active with the pre-adapt
                    # theta snapshot.
                    maml_out = fine_tune(
                        fast_model=maml_model,
                        task=task,
                        mixer_idx=mixer_idx,
                        mixer_name=mixer_name,
                        mixer_outer_params=maml_model.mixer_outer_params(mixer_idx),  # type: ignore[attr-defined]
                        inner_params=maml_model.mixer_inner_params(mixer_idx, anil_mode_effective),  # type: ignore[attr-defined]
                        features=support_x_i,
                        targets=support_targets,
                        holdout_features=holdout_x_i,
                        holdout_targets=holdout_targets,
                        support_coords=support_coords,
                        holdout_coords=holdout_coords,
                        cost_function=cost_function,
                        aux_losses_enabled=aux_losses_enabled,
                        inner_steps=max_steps,
                        inner_lr=fine_tune_lr,
                        inner_optimizer=inner_optimizer,
                        lam=proximal_lam,
                        theta_snapshot=maml_theta_snapshots[mixer_idx],
                        fixed_steps=fixed_steps,
                        max_grad_norm=max_grad_norm,
                        slope_recovery_inner=slope_recovery_inner,
                    )
                    maml_outputs.append(maml_out)

                    # θ₀ branch — no proximal, no theta snapshot.
                    baseline_out = fine_tune(
                        fast_model=baseline_model,
                        task=task,
                        mixer_idx=mixer_idx,
                        mixer_name=mixer_name,
                        mixer_outer_params=baseline_model.mixer_outer_params(mixer_idx),  # type: ignore[attr-defined]
                        inner_params=baseline_model.mixer_inner_params(mixer_idx, anil_mode_effective),  # type: ignore[attr-defined]
                        features=support_x_i,
                        targets=support_targets,
                        holdout_features=holdout_x_i,
                        holdout_targets=holdout_targets,
                        support_coords=support_coords,
                        holdout_coords=holdout_coords,
                        cost_function=cost_function,
                        aux_losses_enabled=aux_losses_enabled,
                        inner_steps=max_steps,
                        inner_lr=fine_tune_lr,
                        inner_optimizer=inner_optimizer,
                        lam=0.0,
                        theta_snapshot=None,
                        fixed_steps=fixed_steps,
                        max_grad_norm=max_grad_norm,
                        slope_recovery_inner=slope_recovery_inner,
                    )
                    baseline_outputs.append(baseline_out)

                # Merge per-mixer outputs into MethodResult instances.
                # build_method_result handles cross-path reconciliation,
                # pct_error, and the per-combo avg_error_pct_per_step.
                maml_method = assemble_method_result(
                    mixer_outputs=maml_outputs,
                    true_coefficients=true_coefficients,
                    fixed_steps=fixed_steps,
                )
                baseline_method = assemble_method_result(
                    mixer_outputs=baseline_outputs,
                    true_coefficients=true_coefficients,
                    fixed_steps=fixed_steps,
                )

                worse = _compute_worse_flags(
                    maml_method, baseline_method, fixed_steps, inner_steps=max_steps
                )

                combo = ComboResult(
                    k=k,
                    noise=noise,
                    maml=maml_method,
                    baseline=baseline_method,
                    worse=worse,
                )

                # Accumulate task-level worse flags (union across combos)
                for step in worse.loss_steps:
                    if step not in task_result.worse.loss_steps:
                        task_result.worse.loss_steps.append(step)
                for coeff_name, steps in worse.coeff_steps.items():
                    if coeff_name not in task_result.worse.coeff_steps:
                        task_result.worse.coeff_steps[coeff_name] = []
                    for step in steps:
                        if step not in task_result.worse.coeff_steps[coeff_name]:
                            task_result.worse.coeff_steps[coeff_name].append(step)

                # Summary — mean across mixers of the final train/holdout loss
                def _final_mean(d: Dict[str, np.ndarray]) -> float:
                    return float(np.mean([arr[-1] for arr in d.values()]))

                maml_train_final = _final_mean(maml_method.fine_tune.per_mixer_train_losses)
                maml_holdout_final = _final_mean(maml_method.fine_tune.per_mixer_holdout_losses)
                baseline_train_final = _final_mean(baseline_method.fine_tune.per_mixer_train_losses)
                baseline_holdout_final = _final_mean(baseline_method.fine_tune.per_mixer_holdout_losses)

                flags: List[str] = []
                if worse.loss_steps:
                    flags.append(
                        f"loss@[{compress_step_ranges(worse.loss_steps, fixed_steps)}]"
                    )
                for coeff_name, steps in worse.coeff_steps.items():
                    if steps:
                        flags.append(
                            f"{coeff_name}@[{compress_step_ranges(steps, fixed_steps)}]"
                        )
                flag_str = f" !!! [{','.join(flags)}]" if flags else ""

                print(
                    f"MAML(train={maml_train_final:.2e}, holdout={maml_holdout_final:.2e}) "
                    f"BL(train={baseline_train_final:.2e}, holdout={baseline_holdout_final:.2e}){flag_str}"
                )

                task_result.combos.append(combo)

            except Exception as e:
                import traceback
                print(f"\n{'=' * 60}")
                print(f"FATAL: {type(e).__name__} during combo K={k}, noise={noise:.0%}:")
                print(f"{'=' * 60}")
                traceback.print_exc()
                print(f"{'=' * 60}")
                print("Fix the issue before running evaluation again.")
                sys.exit(1)

    # ─── Best-combo prediction capture ─────────────────────────────────────
    # TODO(M7.2b): re-enable best-combo prediction capture under the mixer
    # method. The old path called fine_tune(model, features, ...) with an
    # `on_step` callback that snapshotted `model(features)` at each step for
    # spatial visualization. Under the new per-mixer fine_tune, we'd need to
    # (a) re-fine-tune the best combo per-mixer, (b) collect predictions via
    # `forward_one(i, ...)` per mixer per fixed_step, (c) stack them into a
    # (n_steps, holdout, n_outputs) array matching the old BestComboData
    # shape. For now, leave best_combo empty — visualize.py should handle
    # empty BestComboData gracefully (checked via `if bc.combo_key`).
    if task_result.combos:
        # Pick the combo with the lowest overall avg_error_pct (for future
        # best_combo re-fine-tune).
        best = min(
            task_result.combos,
            key=lambda c: (
                min(c.maml.coefficient_recovery.avg_error_pct_per_step)
                if c.maml.coefficient_recovery.avg_error_pct_per_step
                else float("inf")
            ),
        )
        print(f"    Best combo: {best.combo_key} (best_combo prediction capture SKIPPED in M7.2)")

    return task_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MAML vs baseline on meta-test tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to θ* checkpoint (default: final_model.pt in experiment dir)",
    )
    args = parser.parse_args()

    # =========================================================================
    # Load configuration
    # =========================================================================
    print("=" * 60)
    print("MAML Evaluation: θ* vs θ₀ Comparison")
    print("=" * 60)
    print()

    cfg = ExperimentConfig.from_yaml(args.config)
    exp_name = cfg.experiment.name
    base_dir = Path(cfg.output.base_dir)
    exp_dir = base_dir / exp_name

    if not exp_dir.exists():
        # Scan for suffixed directories (ENDNAN → usable, ISNAN → skip)
        candidates = sorted(base_dir.glob(f"{exp_name}-*"))
        endnan = [d for d in candidates if d.name.startswith(f"{exp_name}-ENDNAN")]
        isnan = [d for d in candidates if d.name.startswith(f"{exp_name}-ISNAN")]

        if endnan:
            exp_dir = endnan[0]
            print(f"Using flagged directory: {exp_dir.name}")
        elif isnan:
            print(f"Skipping {exp_name}: flagged as {isnan[0].name}")
            sys.exit(0)
        else:
            print(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)

    # Guard: DONE sentinel means training completed — never overwrite
    # Create samples directory
    eval_dir = exp_dir / "evaluation"
    samples_dir = eval_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    done_path = eval_dir / "DONE"
    if done_path.exists():
        print(f"SKIP: {eval_dir.name} already evaluated (DONE sentinel exists). Delete DONE to overwrite.")
        sys.exit(0)

    # Tee stdout to capture all print output for log file
    _tee = _TeeStream(sys.stdout)
    sys.stdout = _tee  # type: ignore[assignment]

    print(f"Experiment: {exp_name}")
    print(f"Experiment directory: {exp_dir}")
    print()

    # =========================================================================
    # Setup
    # =========================================================================
    seed = cfg.experiment.seed
    device = cfg.experiment.device
    pde_type = cfg.experiment.pde_type

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"PDE type: {pde_type}")
    print()

    # =========================================================================
    # Load checkpoints
    # =========================================================================
    print("-" * 60)
    print("Loading checkpoints...")
    print("-" * 60)

    # θ* (meta-learned) — fall back to latest_model.pt for ENDNAN experiments
    if args.checkpoint is not None:
        theta_star_path = args.checkpoint
    else:
        final_path = exp_dir / "checkpoints" / "final_model.pt"
        best_path = exp_dir / "checkpoints" / "best_model.pt"  # legacy name
        latest_path = exp_dir / "checkpoints" / "latest_model.pt"
        if final_path.exists():
            theta_star_path = final_path
        elif best_path.exists():
            theta_star_path = best_path
            print(f"Using legacy best_model.pt")
        elif latest_path.exists():
            theta_star_path = latest_path
            print(f"final_model.pt not found — using latest_model.pt")
        else:
            print(f"No checkpoint found in {exp_dir / 'checkpoints'}")
            sys.exit(1)

    # θ₀ (initial)
    theta_0_path = exp_dir / "checkpoints" / "initial_model.pt"
    if not theta_0_path.exists():
        raise FileNotFoundError(f"θ₀ checkpoint not found: {theta_0_path}")

    # =========================================================================
    # Load meta-test tasks FIRST — MixerNetwork construction is task-aware so
    # we need any task instance to size the composite network via from_task.
    # =========================================================================
    print("-" * 60)
    print("Loading meta-test tasks...")
    print("-" * 60)

    test_dir = Path(cfg.data.meta_test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Meta-test directory not found: {test_dir}")

    task_class = TASK_REGISTRY.get(pde_type)
    if task_class is None:
        raise ValueError(
            f"Unknown pde_type: {pde_type}. Available: {list(TASK_REGISTRY)}"
        )

    task_pattern = "*_fourier.npz"

    test_loader = MetaLearningDataLoader(
        test_dir, task_class=task_class, task_pattern=task_pattern, device=device
    )
    print()
    print(f"Meta-test tasks: {len(test_loader)}")
    print()

    # =========================================================================
    # Load MixerNetwork checkpoints via the task-aware factory
    # =========================================================================
    # Use any task for architecture sizing — all tasks in the distribution
    # share the same structural feature shape, and the composite network
    # only cares about per-mixer input_dim which comes from
    # task.structural_feature_names.
    sizing_task = test_loader.tasks[0]

    theta_star = load_mixer_from_checkpoint(
        theta_star_path,
        device=device,
        task=sizing_task,
        aux_losses_enabled=cfg.training.aux_losses_enabled,
        hidden_dims=cfg.training.hidden_dims or [350, 350],
        activation=cfg.training.activation or "silu",
        input_bypass=cfg.training.input_bypass,
    )
    theta_0 = load_mixer_from_checkpoint(
        theta_0_path,
        device=device,
        task=sizing_task,
        aux_losses_enabled=cfg.training.aux_losses_enabled,
        hidden_dims=cfg.training.hidden_dims or [350, 350],
        activation=cfg.training.activation or "silu",
        input_bypass=cfg.training.input_bypass,
    )

    print(f"θ* loaded from: {theta_star_path}")
    print(f"θ₀ loaded from: {theta_0_path}")

    # LSLR is MAML-specific; iMAML doesn't use it. Kept as a no-op
    # placeholder so the evaluate_task signature stays symmetric with
    # its old shape.
    lslr_module: Optional[LSLRSchedule] = None

    # iMAML: set up proximal term strength and inner optimizer for evaluation
    proximal_lam = 0.0
    eval_inner_optimizer = "sgd"

    if cfg.training.imaml.enabled:
        im = cfg.training.imaml
        proximal_lam = im.lam

        # Load meta-learned lambda if it was saved by the trainer.
        lam_checkpoint = torch.load(theta_star_path, map_location=device, weights_only=False)
        saved_lam = lam_checkpoint.get("lam")
        if saved_lam is not None:
            # M4b stores self.lam as List[Tensor] (per-mixer). Use the
            # first mixer's lam as the scalar default for the proximal
            # term at eval — per-mixer eval lam isn't plumbed yet, so
            # everyone uses the same lam, which matches how the smoke
            # configs train anyway (lam_lr=0, so both lam[0] and lam[1]
            # stay at their init value and are identical).
            if isinstance(saved_lam, list):
                proximal_lam = float(saved_lam[0].item())
            else:
                proximal_lam = float(saved_lam.item())
            print(f"  Using meta-learned λ={proximal_lam:.4f}")
        else:
            print(f"  Using config λ={proximal_lam}")

        eval_inner_optimizer = im.inner_optimizer
        print(f"  iMAML eval: proximal_lam={proximal_lam}, inner_optimizer={eval_inner_optimizer}")

    print()

    # =========================================================================
    # Evaluation parameters
    # =========================================================================
    ev = cfg.evaluation
    k_values = ev.k_values
    noise_levels = ev.noise_levels
    fine_tune_lr = ev.fine_tune_lr
    max_steps = ev.max_steps
    fixed_steps: List[int] = ev.fixed_steps
    log_weights: bool = ev.log_weights
    loss_type: str = cfg.training.loss_function
    spectral_mode_size: int = cfg.training.spectral_loss.mode_size if cfg.training.spectral_loss.enabled else 0
    max_grad_norm: float = cfg.training.max_grad_norm

    # Build the cost_function used by fine_tune's inner loop. Matches the
    # trainer's `_reset_for_epoch` binding for training-eval consistency —
    # evaluation's closure should see the same objective the training-time
    # closure saw. Spectral loss is not yet supported at eval.
    if spectral_mode_size > 0:
        raise NotImplementedError(
            "Spectral loss is not yet supported in the mixer-method evaluation "
            "pipeline. Set training.spectral_loss.enabled=false for now."
        )

    if loss_type == "mse":
        def _pointwise_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(pred, target)
    elif loss_type == "normalized_mse":
        def _pointwise_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(pred, target) / (target ** 2).mean()
    elif loss_type == "sse":
        def _pointwise_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return ((pred - target) ** 2).sum()
    elif loss_type == "mae":
        def _pointwise_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(pred, target)
    else:
        raise ValueError(
            f"Unknown loss_function: {loss_type}. Use 'mse', 'normalized_mse', 'sse', or 'mae'."
        )

    def cost_function(
        pred: torch.Tensor,
        target: torch.Tensor,
        coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        return _pointwise_loss(pred, target)

    total_combos = len(test_loader) * len(k_values) * len(noise_levels)
    print("-" * 60)
    print("Evaluation grid:")
    print("-" * 60)
    print(f"  K values: {k_values}")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Fine-tune LR: {fine_tune_lr}")
    print(f"  Max steps: {max_steps}")
    print(f"  Fixed steps: {fixed_steps} ({len(fixed_steps)} snapshots)")
    print(f"  Loss function: {loss_type}")
    if max_grad_norm > 0:
        print(f"  Gradient clipping: max_norm={max_grad_norm}")
    print(f"  Total evaluations: {total_combos}")
    print()

    # =========================================================================
    # Run evaluation
    # =========================================================================
    print("=" * 60)
    print("Running evaluation: w/ noisy targets on noise injected holdouts")
    print("=" * 60)
    print()

    holdout_size = ev.holdout_size

    results: Dict[str, Any] = {
        "experiment_name": exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k_values": k_values,
            "noise_levels": noise_levels,
            "fine_tune_lr": fine_tune_lr,
            "max_steps": max_steps,
            "threshold": ev.deriv_threshold,
            "fixed_steps": fixed_steps,
            "holdout_size": holdout_size,
            "pde_type": pde_type,
        },
        "tasks": {},
    }

    # Track per-combo stats by IC type for summary
    combo_stats_by_ic: Dict[str, list] = {}

    for task_idx, task in enumerate(test_loader.tasks):
        print(f"[{task_idx + 1}/{len(test_loader)}] Task: {task.task_name}")

        # Pick the ANIL mode. When ANIL is off, pass "all" so
        # mixer_inner_params returns the whole mixer slice.
        eval_anil_mode = "all"
        if cfg.training.imaml.enabled and cfg.training.imaml.anil:
            eval_anil_mode = cfg.training.imaml.anil_mode

        task_result = evaluate_task(
            task=task,
            theta_star=theta_star,
            theta_0=theta_0,
            k_values=k_values,
            noise_levels=noise_levels,
            fine_tune_lr=fine_tune_lr,
            max_steps=max_steps,
            device=device,
            seed=seed + task_idx * 10000,
            holdout_size=holdout_size,
            fixed_steps=fixed_steps,
            cost_function=cost_function,
            aux_losses_enabled=cfg.training.aux_losses_enabled,
            proximal_lam=proximal_lam,
            inner_optimizer=eval_inner_optimizer,
            anil_mode=eval_anil_mode,
            slope_recovery_inner=(
                cfg.training.imaml.slope_recovery_inner
                if cfg.training.imaml.enabled
                else 0.0
            ),
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            spectral_mode_size=spectral_mode_size,
            log_weights=log_weights,
            lslr=lslr_module,
        )

        # Serialize to NPZ
        task_npz = task_result.to_npz_dict()

        # NaN guard: check all arrays from this task before saving
        nan_keys = [
            k for k, v in task_npz.items()
            if isinstance(v, np.ndarray) and v.dtype.kind == "f" and np.isnan(v).any()
        ]
        if nan_keys:
            print(f"\nFATAL: NaN detected in {len(nan_keys)} arrays for {task.task_name}:")
            for k in nan_keys[:5]:
                print(f"  - {k}")
            if len(nan_keys) > 5:
                print(f"  ... and {len(nan_keys) - 5} more")
            print(f"Cleaning up evaluation directory...")
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
                print(f"Deleted: {eval_dir}")
            sys.exit(1)

        # Save task metadata to results dict
        results["tasks"][task.task_name] = task_result.to_json_dict()

        # Save samples to NPZ
        if task_npz:
            samples_path = samples_dir / f"{task.task_name}.npz"
            np.savez_compressed(samples_path, **task_npz)

        # Collect per-combo stats by IC type (loss + coefficient recovery)
        ic_type = task_result.ic_type
        if ic_type not in combo_stats_by_ic:
            combo_stats_by_ic[ic_type] = []

        for combo in task_result.combos:
            loss_worse_any = len(combo.worse.loss_steps) > 0
            coeff_worse_any = any(
                len(s) > 0 for s in combo.worse.coeff_steps.values()
            )

            # Mean final holdout loss across mixers for the per-IC summary.
            def _final_holdout_mean(mr: MethodResult) -> float:
                per_mixer = mr.fine_tune.per_mixer_holdout_losses
                if not per_mixer:
                    return 0.0
                return float(np.mean([arr[-1] for arr in per_mixer.values()]))

            entry: Dict[str, Any] = {
                "task": task.task_name,
                "k": combo.k,
                "noise": combo.noise,
                "maml_loss": _final_holdout_mean(combo.maml),
                "baseline_loss": _final_holdout_mean(combo.baseline),
                "loss_worse": loss_worse_any,
                "coeff_worse": coeff_worse_any,
            }

            maml_err = combo.maml.coefficient_recovery.avg_error_pct_per_step
            baseline_err = combo.baseline.coefficient_recovery.avg_error_pct_per_step
            if maml_err:
                entry["maml_coeff_error"] = maml_err[-1]
            if baseline_err:
                entry["baseline_coeff_error"] = baseline_err[-1]

            combo_stats_by_ic[ic_type].append(entry)

        print()

    # Save metadata JSON
    results_path = eval_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None if callable(o) else str(o))

    print(f"Saved metadata to: {results_path}")
    print(f"Saved samples to: {samples_dir}")
    print()

    # Print evaluation summary by IC type (loss + coefficient recovery)
    print("-" * 60)
    print("Evaluation Summary by IC Type")
    print("-" * 60)
    for ic_type, entries in sorted(combo_stats_by_ic.items()):
        n = len(entries)
        loss_worse_n = sum(1 for e in entries if e["loss_worse"])
        coeff_worse_n = sum(1 for e in entries if e["coeff_worse"])

        maml_loss_avg = np.mean([e["maml_loss"] for e in entries])
        bl_loss_avg = np.mean([e["baseline_loss"] for e in entries])

        maml_cerrs = [e["maml_coeff_error"] for e in entries if "maml_coeff_error" in e]
        bl_cerrs = [
            e["baseline_coeff_error"] for e in entries if "baseline_coeff_error" in e
        ]

        print(f"\n  {ic_type} ({n} combos):")
        print(
            f"    Loss:  MAML={maml_loss_avg:.2e}  BL={bl_loss_avg:.2e}  MAML worse: {loss_worse_n}/{n}"
        )
        if maml_cerrs:
            print(
                f"    Coeff: MAML={np.mean(maml_cerrs):.1f}%  BL={np.mean(bl_cerrs):.1f}%  MAML worse: {coeff_worse_n}/{n}"
            )

        underperformers = [e for e in entries if e["loss_worse"] or e["coeff_worse"]]
        if underperformers:
            print(f"    Underperforming ({len(underperformers)}/{n}):")
            for e in underperformers[:5]:
                flags = []
                if e["loss_worse"]:
                    flags.append("loss")
                if e["coeff_worse"]:
                    flags.append("coeff")
                print(
                    f"      {e['task']} K={e['k']} noise={e['noise']:.0%} [{','.join(flags)}]"
                )
            if len(underperformers) > 5:
                print(f"      ... and {len(underperformers) - 5} more")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print()
    print(f"Results saved to: {eval_dir}/")
    print("  - results.json (metadata)")
    print("  - samples/*.npz (per-combo arrays)")
    print()
    print("Next steps:")
    print(f"  python scripts/visualize.py --config {args.config}")
    print()

    # Save log
    sys.stdout = _tee.original  # type: ignore[assignment]
    log_path = eval_dir / "evaluate.log"
    log_path.write_text(_tee.getvalue())

    done_path.touch()

    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
