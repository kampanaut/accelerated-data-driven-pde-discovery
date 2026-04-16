"""
iMAML (Implicit Model-Agnostic Meta-Learning) for PDE discovery.

Uses implicit differentiation for the meta-gradient — no unrolled inner loop
graph. Decouples the meta-gradient from the inner optimizer, enabling L-BFGS
or any optimizer in the inner loop.

References:
- Rajeswaran et al. 2019: "Meta-Learning with Implicit Gradients" (NeurIPS)
- Finn et al. 2017: "Model-Agnostic Meta-Learning for Fast Adaptation"
- Raissi 2018: "Deep Hidden Physics Models" (L-BFGS + sin activation)
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ExperimentConfig
import copy
import signal
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task_loader import PDETask, MetaLearningDataLoader
from .spectral_loss import compute_spectral_loss


# ---------------------------------------------------------------------------
# Conjugate Gradient solver (Rajeswaran et al. 2019)
# ---------------------------------------------------------------------------


def cg_solve(
    f_Ax: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    cg_iters: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """
    Conjugate gradient solver for Ax = b.

    Equivalent to minimizing f(x) = 1/2 x^T A x - x^T b.
    A must be positive semi-definite (ensured by λ regularization in iMAML).

    Reference: aravindr93/imaml_dev (Rajeswaran et al. 2019)
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    for i in range(cg_iters):
        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr / (p.dot(Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        if newrdotr < residual_tol:
            break
        beta = newrdotr / (rdotr + 1e-30)
        p = r + beta * p

    return x


def kendall_total_loss(
    fast_model: nn.Module,
    mixer_idx: int,
    features: torch.Tensor,
    targets: torch.Tensor,
    coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    *,
    cost_function: Callable[
        [torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
        torch.Tensor,
    ],
    aux_losses_enabled: bool,
    task: Optional["PDETask"],
) -> torch.Tensor:
    """Main MSE plus Kendall-weighted aux losses for one mixer.

    The single source of truth for the inner-loop objective. Called from
    both the trainer (via `iMAMLTrainer._kendall_total_loss`, which binds
    cost_function / aux_losses_enabled / task from trainer state) and from
    `scripts/evaluate.py`, which needs the same closure at evaluation time
    for training-eval consistency (see plan doc §"Training-evaluation
    consistency").

    Formula per Kendall et al. 2018, eq 7:
        L = ½·exp(-s_mse)·mse + ½·s_mse
            + Σᵢ [½·exp(-s_aux_i)·aux_i + ½·s_aux_i]

    When `aux_losses_enabled` is False, returns just the main MSE — no
    Kendall weighting. The log-variances `s_*` are read from `fast_model`'s
    ParameterDict; they must already be registered (which they are when
    aux_losses_enabled is True and the model was built via
    `MixerNetwork.from_task(..., aux_losses_enabled=True)`).

    Both 0.5 factors in the formula come from the Gaussian NLL derivation:
        -log p(y|f) = (y-f)² / (2σ²) + log σ + const
    with s := log σ² giving `1/(2σ²) = 0.5·exp(-s)` as the weight on L and
    `log σ = 0.5·s` as the regularizer. Kept symmetric per the paper.
    """
    pred = fast_model.forward_one(mixer_idx, features)  # type: ignore[attr-defined]
    # `targets` is stacked shape (N, n_outputs); slice this mixer's column
    # so the shape matches `pred`'s (N,).
    target_i = targets[:, mixer_idx]
    mse = cost_function(pred, target_i, coords)

    if not aux_losses_enabled:
        return mse

    assert task is not None, (
        "task must be provided when aux_losses_enabled=True "
        "(needed for task.auxiliary_losses call)"
    )

    s_mse = fast_model.get_log_sigma(mixer_idx, "mse")  # type: ignore[attr-defined]
    total = 0.5 * torch.exp(-s_mse) * mse + 0.5 * s_mse

    aux_losses = task.auxiliary_losses(mixer_idx, fast_model, features, targets)
    for name, loss in aux_losses.items():
        s_i = fast_model.get_log_sigma(mixer_idx, name)  # type: ignore[attr-defined]
        total = total + 0.5 * torch.exp(-s_i) * loss + 0.5 * s_i

    return total


class iMAMLTrainer:
    """
    iMAML trainer for meta-learning PDE operator initialization.

    Implements implicit MAML (Rajeswaran et al. 2019):
    - Inner loop: optimize task loss + proximal term (SGD or L-BFGS)
    - Meta-gradient: CG solver on implicit Jacobian (no unrolled graph)
    - Outer loop: Adam update on CG-corrected gradients
    - Two-level early stopping: train patience → validation check
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: "ExperimentConfig",
        train_loader: MetaLearningDataLoader,
        val_loader: Optional[MetaLearningDataLoader] = None,
    ):
        """
        Initialize MAML trainer.

        Args:
            model: PDE operator network to meta-train
            cfg: Unified experiment config (training params from cfg.training)
            train_loader: Data loader for meta-training tasks
            val_loader: Optional data loader for meta-validation (early stopping)
        """
        t = cfg.training
        self.config = t  # TrainingSection — all self.config.* reads come from here
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Move model to device and store
        self.device = cfg.experiment.device
        self.model = model.to(self.device)

        # Pointwise loss — used as base term in cost function
        loss_type = t.loss_function
        if loss_type == "mse":
            self._pointwise_loss = F.mse_loss
        elif loss_type == "normalized_mse":
            self._pointwise_loss = lambda pred, target: (
                F.mse_loss(pred, target) / (target**2).mean()
            )
        elif loss_type == "sse":
            self._pointwise_loss = lambda pred, target: ((pred - target) ** 2).sum()
        elif loss_type == "mae":
            self._pointwise_loss = F.l1_loss
        else:
            raise ValueError(
                f"Unknown loss_function: {loss_type}. Use 'mse', 'normalized_mse', 'sse', or 'mae'."
            )

        # Current task domain info — set per-task in compute_task()
        self._current_Lx: float = 0.0
        self._current_Ly: float = 0.0

        # Bind cost function — no conditionals in the hot loop
        self.cost_function = (
            self._spectral_cost
            if t.spectral_loss.enabled
            else self._pointwise_cost
        )

        # iMAML config
        im = t.imaml
        n_outputs: int = model.n_outputs  # type: ignore[attr-defined]
        # Per-mixer lambda — all initialized to the same value, but they
        # diverge independently when lam_lr > 0 (lambda meta-learning).
        self.lam: List[torch.Tensor] = [
            torch.tensor(im.lam, device=self.device, dtype=torch.float32)
            for _ in range(n_outputs)
        ]
        self.lam_lr = im.lam_lr
        self.lam_min = im.lam_min
        self.cg_steps = im.cg_steps
        self.cg_damping = im.cg_damping
        self.slope_recovery_inner = im.slope_recovery_inner

        # Mixer method: Kendall-weighted aux losses (M4a+)
        self.aux_losses_enabled: bool = t.aux_losses_enabled
        # Holds the current task during per-task computation so
        # _kendall_total_loss can call task.auxiliary_losses without
        # threading the task through every inner-loop closure.
        self._current_task: Optional[PDETask] = None
        # Inner-loop parameter list for the CURRENT fast_model. Refreshed
        # once per task at the top of _compute_task_imaml, so the inner
        # solvers don't re-walk model.parameters() on every call. The
        # entries are references to fast_model's Parameter objects (live
        # pointers), so the inner optimizer steps them in place.
        self._current_inner_params: List[nn.Parameter] = []

        # Bind inner solver — SGD (prox every step or at end) or L-BFGS
        if im.inner_optimizer == "lbfgs":
            self._inner_solve = self._inner_solve_lbfgs
        elif im.proximal_every_step:
            self._inner_solve = self._inner_solve_sgd
        else:
            self._inner_solve = self._inner_solve_sgd_prox_end

        # Bind ANIL inner-param selector — one callable per (anil, anil_mode) combo.
        if not im.anil:
            self._inner_params = self._inner_params_all
        elif im.anil_mode == "head":
            self._inner_params = self._inner_params_head
        elif im.anil_mode == "head+scales_all":
            self._inner_params = self._inner_params_head_scales_all
        elif im.anil_mode == "head+scales_last":
            self._inner_params = self._inner_params_head_scales_last
        else:
            raise ValueError(
                f"Unknown anil_mode={im.anil_mode!r}. "
                f"Use 'head', 'head+scales_all', or 'head+scales_last'."
            )

        prox_mode = "every_step" if im.proximal_every_step else "end_only"
        anil_str = f", ANIL ({im.anil_mode})" if im.anil else ""
        print(f"  iMAML: lam={im.lam}, cg_steps={im.cg_steps}, "
              f"cg_damping={im.cg_damping}, inner_optimizer={im.inner_optimizer}, "
              f"proximal={prox_mode}{anil_str}")

        # Outer loop optimizer
        self._reset_for_epoch()

        # Checkpoint mode validation: exactly one of patience or checkpoint_interval
        if t.patience > 0 and t.checkpoint_interval == 0:
            self._patience_enabled = True
        elif t.patience == 0 and t.checkpoint_interval > 0:
            self._patience_enabled = False
        else:
            raise ValueError(
                f"Invalid checkpoint config: patience={t.patience}, "
                f"checkpoint_interval={t.checkpoint_interval}. "
                f"Use (patience>0, checkpoint_interval=0) for early stopping, "
                f"or (patience=0, checkpoint_interval>0) for periodic saves."
            )

        # Mutual exclusion: iMAML is incompatible with MSL and LSLR
        if t.msl_enabled:
            raise ValueError("MSL is MAML-specific (per-step outer loss), incompatible with iMAML.")
        if t.lslr_enabled:
            raise ValueError("LSLR is MAML-specific (per-step learnable LRs), incompatible with iMAML.")

        # L-BFGS + inner_lr validation
        if im.inner_optimizer == "lbfgs" and t.inner_lr != 0.01:
            raise ValueError(
                f"inner_lr={t.inner_lr} is set but inner_optimizer=lbfgs ignores it. "
                f"Set inner_lr=0.01 (default) or use inner_optimizer=sgd."
            )

        # L-BFGS + grad clipping warning (clipping only applies to outer step, not inner)
        if im.inner_optimizer == "lbfgs" and t.max_grad_norm > 0:
            print(f"  Note: max_grad_norm={t.max_grad_norm} applies to outer step only. "
                  f"L-BFGS inner loop uses line search instead of gradient clipping.")

        # DA → CG-step annealing: start with cg_steps=0, switch to cg_steps after da_threshold
        self._cg_steps_active = im.cg_steps
        if t.da_enabled:
            self._cg_steps_active = 0  # Phase 1: FOMAML equivalent
            print(f"  DA enabled: CG=0 until iter {t.da_threshold}, then CG={im.cg_steps}")
        elif t.first_order:
            self._cg_steps_active = 0  # Permanent FOMAML equivalent
            print(f"  first_order=True: CG=0 (FOMAML equivalent)")

        # Bind compute_task
        self.compute_task = self._compute_task_imaml

        # Bind iteration hook + finalize — no conditionals in the hot loop
        if self._patience_enabled:
            self._iteration_hook = self._patience_iteration_hook
            self._training_finalize = self._patience_finalize
        else:
            self._iteration_hook = self._interval_iteration_hook
            self._training_finalize = self._interval_finalize

        # Training state
        self.iteration = 0
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_train_state = None  # Stash weights when train loss improves (patience mode only)
        self.patience_counter = 0
        self._nan_at: Optional[int] = None
        self._last_train_loss: float = 0.0
        self._early_stopped: bool = False
        self.history: Dict[str, Any] = self._fresh_history()
        # Side-channel: populated inside _compute_meta_gradient each iter and
        # appended to history by _run_phase after the NaN check, so history
        # lists stay length-aligned with train_loss/iteration on NaN rollback.
        self._last_iter_metrics: Optional[Dict[str, Any]] = None
        self._resumed = False
        self._stop_requested = False


    def _reset_for_epoch(self) -> None:
        """Reset per-mixer optimizers + schedulers for a new epoch. Model weights unchanged.

        Every field is a list of length `n_outputs`. For a 1-output PDE the
        lists are length 1 and behavior matches the legacy one-head path.
        For 2-output PDEs each mixer owns an independent optimizer and
        scheduler over its own parameter slice (body + head + its own
        Kendall log-variances), with no shared state across mixers.
        """
        t = self.config
        im = t.imaml
        n_outputs: int = self.model.n_outputs  # type: ignore[attr-defined]

        # Per-mixer parameter lists. Partition is exact and disjoint:
        # union(mixer_parameters(i) + mixer_log_variance_parameters(i))
        # == list(self.model.parameters()).
        self._opt_params_list: List[List[nn.Parameter]] = []
        for i in range(n_outputs):
            params_i = (
                self.model.mixer_parameters(i)  # type: ignore[attr-defined]
                + self.model.mixer_log_variance_parameters(i)  # type: ignore[attr-defined]
            )
            self._opt_params_list.append(params_i)

        # Build per-mixer outer optimizers. All mixers get the same
        # optimizer type (Adam/LBFGS/adam+lbfgs) — the branch picks the
        # type once, then instantiates `n_outputs` copies.
        self.outer_opts: List[torch.optim.Optimizer] = []
        if im.outer_optimizer == "lbfgs":
            for params_i in self._opt_params_list:
                self.outer_opts.append(torch.optim.LBFGS(
                    params_i, lr=1.0, max_iter=1, max_eval=20,
                    tolerance_grad=1e-15, tolerance_change=1e-15,
                    line_search_fn="strong_wolfe",
                ))
            self._outer_step = self._outer_step_lbfgs
            print(f"  Outer optimizer: {n_outputs}× L-BFGS (no LR, no scheduler needed)")
        elif im.outer_optimizer == "adam+lbfgs":
            for params_i in self._opt_params_list:
                self.outer_opts.append(torch.optim.Adam(
                    params_i, lr=t.outer_lr, betas=tuple(t.adam_betas), eps=1e-3,
                ))
            self._outer_step = self._outer_step_adam
            self._outer_lbfgs_after = im.outer_lbfgs_after
            print(f"  Outer optimizer: {n_outputs}× Adam → {n_outputs}× L-BFGS after iter {im.outer_lbfgs_after}")
        else:
            for params_i in self._opt_params_list:
                self.outer_opts.append(torch.optim.Adam(
                    params_i, lr=t.outer_lr, betas=tuple(t.adam_betas), eps=1e-3,
                ))  # eps=1e-3 matches reference (aravindr93/imaml_dev, learner_model.py:17)
            self._outer_step = self._outer_step_adam
            print(f"  Outer optimizer: {n_outputs}× Adam")

        # LR scheduler per optimizer. Each mixer gets its own scheduler
        # chain so phase transitions (warmup, cosine, plateau, etc.) are
        # independent across mixers.
        self.schedulers: List[Optional[torch.optim.lr_scheduler.LRScheduler]] = [None] * n_outputs
        if t.has_scheduler:
            for i, opt_i in enumerate(self.outer_opts):
                self.schedulers[i] = self._build_scheduler_for(opt_i)

        # Plateau scheduler state (shared across all mixers — same task-loss signal)
        self._plateau_mode = (t.use_scheduler and t.scheduler_type == "plateau")
        self._loss_window: List[float] = []

    def _build_scheduler_for(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Build one scheduler chain (warmup → decay) for a single optimizer.

        Mirrors the legacy single-optimizer scheduler construction; the
        only change is that each mixer's optimizer gets its own chain.
        """
        t = self.config
        schedulers: List[torch.optim.lr_scheduler.LRScheduler] = []
        milestones: List[int] = []

        # Phase 1: linear warmup
        if t.warmup_iterations > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / t.warmup_iterations,
                end_factor=1.0,
                total_iters=t.warmup_iterations,
            )
            schedulers.append(warmup)
            milestones.append(t.warmup_iterations)

        # Phase 2: decay
        if t.use_scheduler:
            post_warmup_iters = t.max_iterations - t.warmup_iterations
            if t.scheduler_type == "warm_restarts":
                decay = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=t.T_0, T_mult=t.T_mult, eta_min=t.min_lr,
                )
            elif t.scheduler_type == "exponential":
                gamma = (t.min_lr / t.outer_lr) ** (1.0 / post_warmup_iters)
                decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            elif t.scheduler_type == "polynomial":
                decay = torch.optim.lr_scheduler.PolynomialLR(
                    optimizer, total_iters=post_warmup_iters, power=t.poly_power,
                )
            elif t.scheduler_type == "plateau":
                decay = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=t.plateau_factor,
                    patience=t.plateau_patience,
                    threshold=t.plateau_threshold,
                    cooldown=t.plateau_cooldown,
                    min_lr=t.min_lr,
                )
            else:
                decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=post_warmup_iters, eta_min=t.min_lr,
                )
            schedulers.append(decay)  # type: ignore[arg-type]

        if len(schedulers) > 1:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones,
            )
        if schedulers:
            return schedulers[0]
        return None

    # ------------------------------------------------------------------
    # Cost function: pointwise MSE + optional spectral loss
    # ------------------------------------------------------------------

    def _pointwise_cost(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coords: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Pointwise-only cost (no spectral)."""
        return self._pointwise_loss(pred, target)

    def _spectral_cost(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coords: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Pointwise + spectral cost."""
        pw = self._pointwise_loss(pred, target)
        if coords is None:
            return pw
        x_pts, y_pts = coords
        spec = compute_spectral_loss(
            pred,
            target,
            x_pts,
            y_pts,
            self._current_Lx,
            self._current_Ly,
            self.config.spectral_loss.mode_size,
        )
        return pw + spec

    # ------------------------------------------------------------------
    # Parameter vector helpers
    # ------------------------------------------------------------------

    # ANIL inner-param selectors (one is bound to self._inner_params in __init__).
    # Full ANIL off → everything. ANIL on → head, optionally plus adaptive scales.

    @staticmethod
    def _kendall_param_ids(model: nn.Module) -> set[int]:
        """Return id()s of Kendall log-variance parameters (if any).

        Used by the inner-params selectors to exclude log-variances from
        the inner-loop adapted set — log-variances are outer-only
        meta-parameters and must never be adapted by the inner loop,
        regardless of ANIL mode.
        """
        log_vars = getattr(model, "log_variances", None)
        if log_vars is None:
            return set()
        return {id(p) for p in log_vars.parameters()}

    def _concat_mixer_inner_params(
        self, model: nn.Module, anil_mode: str
    ) -> List[nn.Parameter]:
        """Concatenate `mixer_inner_params(i, mode)` across all mixers.

        Thin wrapper over the composite's per-mixer getter. The union
        equals the composite's inner-adapted set for that ANIL mode with
        Kendall log-variances excluded by construction (log-variances
        live on the composite, not inside individual mixers).
        """
        n_outputs: int = getattr(model, "n_outputs", 1)
        params: List[nn.Parameter] = []
        for i in range(n_outputs):
            params.extend(model.mixer_inner_params(i, anil_mode))  # type: ignore[attr-defined]
        return params

    def _inner_params_all(self, model: nn.Module) -> List[nn.Parameter]:
        return self._concat_mixer_inner_params(model, "all")

    def _inner_params_head(self, model: nn.Module) -> List[nn.Parameter]:
        return self._concat_mixer_inner_params(model, "head")

    def _inner_params_head_scales_all(self, model: nn.Module) -> List[nn.Parameter]:
        return self._concat_mixer_inner_params(model, "head+scales_all")

    def _inner_params_head_scales_last(self, model: nn.Module) -> List[nn.Parameter]:
        return self._concat_mixer_inner_params(model, "head+scales_last")

    def _get_flat_params(self, params: List[nn.Parameter]) -> torch.Tensor:
        """Return a flattened, detached-clone view of the given parameter list."""
        return torch.cat([p.data.view(-1) for p in params]).clone()

    def _set_flat_params(
        self, params: List[nn.Parameter], flat: torch.Tensor
    ) -> None:
        """Write a flattened vector back into the given parameter list."""
        offset = 0
        for p in params:
            n = p.nelement()
            p.data.copy_(flat[offset:offset + n].view(p.size()))
            offset += n

    def _regularization_loss(
        self,
        params: List[nn.Parameter],
        theta: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        """Proximal term: λ/2 ||φ - θ||² (Eq. 3 in Rajeswaran et al. 2019).

        Operates on a parameter list, not a model, so per-mixer inner loops
        can pass only their own mixer's full slice. `theta` must be a flat
        vector sized to match the concatenated `params`.
        """
        phi = torch.cat([p.view(-1) for p in params])
        return 0.5 * lam * (phi - theta).pow(2).sum()

    def _kendall_total_loss(
        self,
        fast_model: nn.Module,
        mixer_idx: int,
        features: torch.Tensor,
        targets: torch.Tensor,
        coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Trainer-side thin wrapper over the module-level `kendall_total_loss`.

        Binds the trainer's `cost_function`, `aux_losses_enabled` flag, and
        currently-bound `_current_task` into the free module function so the
        inner-loop closures can call `self._kendall_total_loss(...)` without
        passing those three every time. The free function is the single
        source of truth for the formula — `scripts/evaluate.py` imports it
        directly and passes its own cost_function / aux flag / task.
        """
        return kendall_total_loss(
            fast_model, mixer_idx, features, targets, coords,
            cost_function=self.cost_function,
            aux_losses_enabled=self.aux_losses_enabled,
            task=self._current_task,
        )

    def _fresh_history(self) -> Dict[str, Any]:
        """Build an empty expanded-history dict matching the M5a schema.

        Per-mixer sub-dicts carry pre-allocated empty lists for every field
        that will be appended during training. Aux-related fields are empty
        dicts when `aux_losses_enabled=false` — no appending happens under
        that flag, so their emptiness is a post-hoc "not applicable" marker.
        """
        n_outputs: int = self.model.n_outputs  # type: ignore[attr-defined]
        aux_names_per_mixer: List[List[str]] = (
            self.train_loader.tasks[0].aux_loss_names  # type: ignore[attr-defined]
        )
        mixer_history: Dict[str, Any] = {}
        for i in range(n_outputs):
            mixer_i: Dict[str, Any] = {
                "mse_main": [],
                "aux": {name: [] for name in aux_names_per_mixer[i]}
                if self.aux_losses_enabled
                else {},
                "s_mse": [],
                "s_aux": {name: [] for name in aux_names_per_mixer[i]}
                if self.aux_losses_enabled
                else {},
                "eff_weight_mse": [],
                "eff_weight_aux": {name: [] for name in aux_names_per_mixer[i]}
                if self.aux_losses_enabled
                else {},
                "support_pre_adapt": [],
                "support_post_adapt": [],
            }
            mixer_history[str(i)] = mixer_i
        return {
            "iteration": [],
            "train_loss": [],
            "val_loss": [],
            "lam": [],   # list of per-iter [lam_0, lam_1, ...] snapshots
            "lr": [],    # list of per-iter [lr_0, lr_1, ...] snapshots
            "mixers": mixer_history,
        }

    def _compute_raw_losses(
        self,
        fast_model: nn.Module,
        mixer_idx: int,
        features: torch.Tensor,
        targets: torch.Tensor,
        coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[float, Dict[str, float]]:
        """Raw unweighted mse_main + per-name aux loss scalars for logging.

        Returns (mse_main_scalar, aux_dict). Detached scalars — no autograd
        graph retained after the function returns. Intended for history
        logging only, not for optimization. One extra forward pass per call
        (plus one aux-loss computation when aux_losses_enabled is true).
        """
        pred = fast_model.forward_one(mixer_idx, features)  # type: ignore[attr-defined]
        target_i = targets[:, mixer_idx]
        mse_scalar = self.cost_function(pred, target_i, coords).item()

        aux_raw: Dict[str, float] = {}
        if self.aux_losses_enabled and self._current_task is not None:
            aux_tensors = self._current_task.auxiliary_losses(
                mixer_idx, fast_model, features, targets
            )
            for name, loss in aux_tensors.items():
                aux_raw[name] = loss.item()

        return mse_scalar, aux_raw

    # ------------------------------------------------------------------
    # Inner solvers (bound in __init__)
    # ------------------------------------------------------------------

    def _inner_solve_sgd(
        self,
        fast_model: nn.Module,
        mixer_idx: int,
        mixer_outer_params: List[nn.Parameter],
        theta: torch.Tensor,
        lam: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Inner loop: SGD on inner_cost + proximal for inner_steps steps (Eq. 3)."""
        opt = torch.optim.SGD(self._current_inner_params, lr=self.config.inner_lr)
        for _ in range(self.config.inner_steps):
            opt.zero_grad()
            task_loss = self._kendall_total_loss(
                fast_model, mixer_idx, support_x, support_y, support_coords
            )
            if self.slope_recovery_inner > 0:
                task_loss = task_loss + self.slope_recovery_inner * fast_model.mixer_slope_recovery(mixer_idx)  # type: ignore[attr-defined]
            prox = self._regularization_loss(mixer_outer_params, theta, lam)
            total = task_loss + prox
            total.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(mixer_outer_params, self.config.max_grad_norm)
            opt.step()

    def _inner_solve_sgd_prox_end(
        self,
        fast_model: nn.Module,
        mixer_idx: int,
        mixer_outer_params: List[nn.Parameter],
        theta: torch.Tensor,
        lam: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Inner loop: plain SGD with inner_cost then one proximal step at end."""
        opt = torch.optim.SGD(self._current_inner_params, lr=self.config.inner_lr)
        for _ in range(self.config.inner_steps):
            opt.zero_grad()
            task_loss = self._kendall_total_loss(
                fast_model, mixer_idx, support_x, support_y, support_coords
            )
            if self.slope_recovery_inner > 0:
                task_loss = task_loss + self.slope_recovery_inner * fast_model.mixer_slope_recovery(mixer_idx)  # type: ignore[attr-defined]
            task_loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(mixer_outer_params, self.config.max_grad_norm)
            opt.step()
        # One proximal pull-back step
        opt.zero_grad()
        prox = self._regularization_loss(mixer_outer_params, theta, lam)
        prox.backward()
        opt.step()

    def _inner_solve_lbfgs(
        self,
        fast_model: nn.Module,
        mixer_idx: int,
        mixer_outer_params: List[nn.Parameter],
        theta: torch.Tensor,
        lam: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Inner loop: L-BFGS on task_loss + proximal (Raissi 2018 style)."""
        opt = torch.optim.LBFGS(
            self._current_inner_params,
            lr=1.0,
            max_iter=self.config.inner_steps,
            tolerance_grad=1e-15,
            tolerance_change=1e-15,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            task_loss = self._kendall_total_loss(
                fast_model, mixer_idx, support_x, support_y, support_coords
            )
            if self.slope_recovery_inner > 0:
                task_loss = task_loss + self.slope_recovery_inner * fast_model.mixer_slope_recovery(mixer_idx)  # type: ignore[attr-defined]
            prox = self._regularization_loss(mixer_outer_params, theta, lam)
            total = task_loss + prox
            total.backward()
            return total

        opt.step(closure)

    # ------------------------------------------------------------------
    # Hessian-vector product and CG matrix evaluator
    # ------------------------------------------------------------------

    def _hessian_vector_product(
        self,
        model: nn.Module,
        mixer_idx: int,
        mixer_outer_params: List[nn.Parameter],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute H·v where H = ∇²φ_i L̂_i(φ_i) for one mixer.

        Uses double autograd.grad trick (Rajeswaran et al. 2019). The
        Hessian is of mixer `mixer_idx`'s task loss (Kendall-weighted
        total plus inner slope recovery if enabled), taken with respect
        to that mixer's full outer-parameter slice. The proximal term's
        Hessian is λI, already accounted for in the matrix evaluator.
        """
        inner_obj = self._kendall_total_loss(
            model, mixer_idx, support_x, support_y, support_coords
        )
        if self.slope_recovery_inner > 0:
            inner_obj = inner_obj + self.slope_recovery_inner * model.mixer_slope_recovery(mixer_idx)  # type: ignore[attr-defined]
        grad_ft = torch.autograd.grad(inner_obj, mixer_outer_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        h = torch.sum(flat_grad * vector)
        hvp = torch.autograd.grad(h, mixer_outer_params)
        return torch.cat([g.contiguous().view(-1) for g in hvp])

    def _matrix_evaluator(
        self,
        model: nn.Module,
        mixer_idx: int,
        mixer_outer_params: List[nn.Parameter],
        lam: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build CG operator A_i for mixer `mixer_idx`:
        A_i v = (1 + regu_coef) * v + H_i v / (lam_i + lam_damping).

        Matches reference (aravindr93/imaml_dev, learner_model.py:143).
        regu_coef = cg_damping (default 1.0), lam_damping hardcoded at 10.0.
        Per-mixer lambda lets each mixer pull back at its own strength
        when lam_lr > 0 makes lambda learnable.
        """
        regu_coef = self.cg_damping
        lam_damping = 10.0  # hardcoded in reference

        def evaluator(v: torch.Tensor) -> torch.Tensor:
            hvp = self._hessian_vector_product(
                model, mixer_idx, mixer_outer_params,
                support_x, support_y, support_coords, v,
            )
            return (1.0 + regu_coef) * v + hvp / (lam + lam_damping)

        return evaluator

    # ------------------------------------------------------------------
    # Core iMAML task computation
    # ------------------------------------------------------------------

    def _task_setup(
        self, task: PDETask, seed: int
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor,
               Optional[Tuple[torch.Tensor, torch.Tensor]],
               Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Common setup: split into support/query, return per-mixer feature lists.

        `support_x_list[i]` and `query_x_list[i]` are the structural feature
        tensors for mixer `i`. For 1-output PDEs the lists are length 1.

        When `training.noise_augmentation.enabled` is true, samples a per-call
        `noise_level ~ Uniform(range[0], range[1])` with a seeded generator and
        builds a device-side generator for the Gaussian noise injection inside
        `task.inject_noise_at_source`. Both generators are seeded off `seed`
        with disjoint streams so the whole training run is reproducible.
        """
        noise_level = 0.0
        noise_generator: Optional[torch.Generator] = None
        if self.config.noise_augmentation.enabled:
            lo, hi = self.config.noise_augmentation.range
            # Separate streams: one CPU generator for the noise_level scalar,
            # one device generator for the Gaussian tensor in inject_noise_at_source.
            # Both derived from `seed` with disjoint offsets for determinism.
            level_gen = torch.Generator(device="cpu").manual_seed(seed + 0x01)
            u01 = torch.rand(1, generator=level_gen).item()
            noise_level = float(lo + u01 * (hi - lo))
            noise_generator = torch.Generator(device=self.device).manual_seed(
                seed + 0x02
            )

        support, query, support_coords, query_coords = task.get_support_query_split(
            K_shot=self.config.k_shot,
            query_size=self.config.query_size,
            k_seed=seed,
            noise_level=noise_level,
            noise_generator=noise_generator,
        )

        if self.config.inner_steps == 0:
            raise ValueError("You cannot run training with inner_steps == 0")

        support_x_list, support_y = support
        query_x_list, query_y = query

        self._current_Lx = task.Lx
        self._current_Ly = task.Ly

        return support_x_list, support_y, query_x_list, query_y, support_coords, query_coords

    def _compute_task_imaml(
        self, task: PDETask, seed: int
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        float,
        List[Dict[str, Any]],
    ]:
        """One task, generalized over n_outputs mixers.

        Returns a 4-tuple:
          - `corrected_grads`: list of CG-corrected meta-gradients, length n_outputs
          - `train_grads`: list of per-mixer train gradients, populated only
            when `lam_lr > 0` (empty otherwise)
          - `avg_query_loss`: scalar, mean query loss across mixers
          - `task_metrics`: list of dicts, length n_outputs, holding per-mixer
            logging scalars for this task: `support_pre_adapt`,
            `support_post_adapt`, `mse_main` (raw, unweighted), `aux` (dict of
            raw aux-loss scalars). These are averaged across the meta-batch's
            tasks by `_compute_meta_gradient` before appending to history.

        Each mixer's iMAML chain runs independently:
          1. Snapshot per-mixer theta (full mixer slice)
          2. Inner solve adapts mixer's ANIL subset
          3. Query loss + query gradient over mixer's full slice
          4. HVP/CG correction over mixer's full slice
          5. (Only when lam_lr > 0) train gradient at φ* over mixer's slice,
             reused by the lambda meta-learning step in _compute_meta_gradient.
        """
        # Make the task visible to _kendall_total_loss without threading it
        # through every inner-loop closure signature.
        self._current_task = task

        support_x_list, support_y, query_x_list, query_y, support_coords, query_coords = (
            self._task_setup(task, seed)
        )

        n_outputs: int = self.model.n_outputs  # type: ignore[attr-defined]
        anil_mode = self.config.imaml.anil_mode if self.config.imaml.anil else "all"

        # Per-mixer pre-adapt theta snapshots, taken on the LIVE model
        # before deepcopy. Each snapshot is the full mixer slice.
        theta_list: List[torch.Tensor] = [
            self._get_flat_params(
                self.model.mixer_outer_params(i)  # type: ignore[attr-defined]
            )
            for i in range(n_outputs)
        ]

        fast_model = copy.deepcopy(self.model)

        # Per-mixer fast-model slices and inner-adapted subsets
        fast_outer: List[List[nn.Parameter]] = [
            fast_model.mixer_outer_params(i)  # type: ignore[attr-defined]
            for i in range(n_outputs)
        ]
        fast_inner: List[List[nn.Parameter]] = [
            fast_model.mixer_inner_params(i, anil_mode)  # type: ignore[attr-defined]
            for i in range(n_outputs)
        ]

        verbose = not getattr(sys.stdout, 'quiet', False)

        corrected_grads: List[torch.Tensor] = []
        train_grads: List[torch.Tensor] = []
        task_metrics: List[Dict[str, Any]] = []
        query_loss_total = 0.0

        for mixer_idx in range(n_outputs):
            mixer_outer = fast_outer[mixer_idx]
            theta_i = theta_list[mixer_idx]
            lam_i = self.lam[mixer_idx]
            support_x_i = support_x_list[mixer_idx]
            query_x_i = query_x_list[mixer_idx]

            # Reassign the current inner-params cache to this mixer's slice
            # so the bound _inner_solve_* method picks up the right list.
            self._current_inner_params = fast_inner[mixer_idx]

            # Pre-adapt support loss — always computed (was verbose-only),
            # captured for logging. No no_grad() wrapper: _kendall_total_loss
            # may call task.auxiliary_losses which uses torch.autograd.grad
            # with create_graph=True internally and fails inside no_grad.
            pre_loss_t = self._kendall_total_loss(
                fast_model, mixer_idx, support_x_i, support_y, support_coords
            )
            pre_loss_val = pre_loss_t.item()
            del pre_loss_t
            if verbose:
                print(f"\t\tmixer {mixer_idx} pre-adapt: support_loss={pre_loss_val:.6f}")

            # Inner solve: minimize task_loss_i + λ_i/2 ||φ_i - θ_i||²
            self._inner_solve(
                fast_model, mixer_idx, mixer_outer, theta_i, lam_i,
                support_x_i, support_y, support_coords,
            )

            # Post-adapt support loss + raw unweighted breakdown for logging.
            post_loss_t = self._kendall_total_loss(
                fast_model, mixer_idx, support_x_i, support_y, support_coords
            )
            post_loss_val = post_loss_t.item()
            del post_loss_t
            mse_raw, aux_raw = self._compute_raw_losses(
                fast_model, mixer_idx, support_x_i, support_y, support_coords
            )
            task_metrics.append({
                "support_pre_adapt": pre_loss_val,
                "support_post_adapt": post_loss_val,
                "mse_main": mse_raw,
                "aux": aux_raw,
            })
            if verbose:
                print(f"\t\tmixer {mixer_idx} post-adapt: support_loss={post_loss_val:.6f}"
                      f" ({self.config.inner_steps} steps)")

            # Query gradient: v_i = ∇_{φ_i} L_query(φ_i*) over mixer_i's full slice
            query_loss_i = self._kendall_total_loss(
                fast_model, mixer_idx, query_x_i, query_y, query_coords
            )
            query_grad_i = torch.autograd.grad(query_loss_i, mixer_outer)
            flat_grad_i = torch.cat([g.contiguous().view(-1) for g in query_grad_i])
            query_loss_val_i = query_loss_i.item()
            query_loss_total += query_loss_val_i

            # CG correction: g_i = (I + 1/λ_i · H_i)⁻¹ v_i
            cg = self._cg_steps_active
            if cg <= 1:
                corrected_grad_i = flat_grad_i.detach()
            else:
                evaluator = self._matrix_evaluator(
                    fast_model, mixer_idx, mixer_outer, lam_i,
                    support_x_i, support_y, support_coords,
                )
                corrected_grad_i = cg_solve(evaluator, flat_grad_i.detach(), cg)

            corrected_grads.append(corrected_grad_i)

            # Lambda meta-gradient prep — match reference (omniglot_implicit_maml.py):
            # compute train_grad on the SAME fast_model that just adapted, so the
            # lam_lr branch in _compute_meta_gradient doesn't redo the inner solve.
            if self.lam_lr > 0:
                train_loss_i = self._kendall_total_loss(
                    fast_model, mixer_idx, support_x_i, support_y, support_coords
                )
                train_grad_t = torch.autograd.grad(train_loss_i, mixer_outer)
                flat_train_grad = torch.cat(
                    [g.contiguous().view(-1) for g in train_grad_t]
                ).detach()
                train_grads.append(flat_train_grad)

            if verbose:
                print(f"\t\t\tmixer {mixer_idx} loss={query_loss_val_i:.2f}")

        avg_query_loss = query_loss_total / n_outputs
        return corrected_grads, train_grads, avg_query_loss, task_metrics

    def _compute_meta_gradient(self, tasks: List[PDETask]) -> float:
        """Compute per-mixer CG-corrected meta-gradients and set .grad on params.

        Each mixer accumulates its own meta-gradient vector across tasks; at
        the end of the meta-batch, that vector is written into the mixer's
        parameter `.grad` attributes. The per-mixer outer optimizers then
        each step their own slice with no crosstalk.

        Returns average query loss across tasks (averaged over mixers).
        Does NOT call optimizer.step().
        """
        self.model.train()

        n_outputs: int = self.model.n_outputs  # type: ignore[attr-defined]

        # One meta-gradient accumulator per mixer, sized to that mixer's
        # full outer slice (body + head + log-vars when aux is on).
        meta_grads: List[torch.Tensor] = [
            torch.zeros(
                sum(p.numel() for p in self._opt_params_list[i]),
                device=self.device,
                dtype=self._opt_params_list[i][0].dtype,
            )
            for i in range(n_outputs)
        ]
        total_loss = 0.0
        self._lam_grad: List[float] = [0.0] * n_outputs

        # Per-mixer accumulators for M5a logging fields (raw loss scalars,
        # pre/post adapt support losses). Summed across the batch then
        # averaged at the end.
        acc_support_pre: List[float] = [0.0] * n_outputs
        acc_support_post: List[float] = [0.0] * n_outputs
        acc_mse_main: List[float] = [0.0] * n_outputs
        acc_aux: List[Dict[str, float]] = [{} for _ in range(n_outputs)]

        for task_idx, task in enumerate(tasks):
            print(f"\ttask [{task_idx}/{len(tasks)}]: {task.task_name}")
            task_seed = self.iteration * len(tasks) + task_idx
            corrected_grads, train_grads, loss_val, task_metrics = self.compute_task(task, task_seed)
            for i in range(n_outputs):
                meta_grads[i] += corrected_grads[i] / len(tasks)
                acc_support_pre[i] += task_metrics[i]["support_pre_adapt"]
                acc_support_post[i] += task_metrics[i]["support_post_adapt"]
                acc_mse_main[i] += task_metrics[i]["mse_main"]
                for name, aux_val in task_metrics[i]["aux"].items():
                    acc_aux[i][name] = acc_aux[i].get(name, 0.0) + aux_val
            total_loss += loss_val / len(tasks)

            # Per-mixer lambda meta-learning. train_grads is populated by
            # _compute_task_imaml only when self.lam_lr > 0 (no extra inner
            # solve — fast_model from the main path was reused). Matches the
            # reference (omniglot_implicit_maml.py:131-141).
            if self.lam_lr > 0:
                for i in range(n_outputs):
                    inner_prod = train_grads[i].dot(corrected_grads[i])
                    task_lam_grad = (inner_prod / (self.lam[i] ** 2 + 0.1)).item()
                    self._lam_grad[i] += task_lam_grad / len(tasks)

        avg_loss = total_loss
        if not torch.tensor(avg_loss).isfinite():
            self._last_iter_metrics = None
            return float("nan")

        # Build the per-iter logging blob. Averaged across the batch's tasks,
        # plus end-of-iter snapshots of log-variances / effective weights /
        # λ / LR. Appended to history by _run_phase after the NaN check so
        # history lists stay length-aligned with train_loss on NaN rollback.
        n_tasks = len(tasks)
        mixer_blobs: Dict[str, Dict[str, Any]] = {}
        for i in range(n_outputs):
            mixer_i: Dict[str, Any] = {
                "support_pre_adapt": acc_support_pre[i] / n_tasks,
                "support_post_adapt": acc_support_post[i] / n_tasks,
                "mse_main": acc_mse_main[i] / n_tasks,
                "aux": {name: v / n_tasks for name, v in acc_aux[i].items()},
            }
            if self.aux_losses_enabled:
                # Snapshot Kendall log-variances for this mixer.
                s_mse_t = self.model.get_log_sigma(i, "mse")  # type: ignore[attr-defined]
                mixer_i["s_mse"] = s_mse_t.item()
                mixer_i["eff_weight_mse"] = torch.exp(-s_mse_t).item()
                s_aux_map: Dict[str, float] = {}
                eff_aux_map: Dict[str, float] = {}
                for name in acc_aux[i].keys():
                    s_aux_t = self.model.get_log_sigma(i, name)  # type: ignore[attr-defined]
                    s_aux_map[name] = s_aux_t.item()
                    eff_aux_map[name] = torch.exp(-s_aux_t).item()
                mixer_i["s_aux"] = s_aux_map
                mixer_i["eff_weight_aux"] = eff_aux_map
            mixer_blobs[str(i)] = mixer_i

        self._last_iter_metrics = {
            "lam": [l.item() for l in self.lam],
            "lr": [opt_i.param_groups[0]["lr"] for opt_i in self.outer_opts],
            "mixers": mixer_blobs,
        }

        # Per-mixer .grad writeback. For each mixer, walk its outer slice
        # and copy the corresponding meta_grad slice into p.grad.
        for i in range(n_outputs):
            offset = 0
            for p in self._opt_params_list[i]:
                numel = p.numel()
                p.grad = meta_grads[i][offset:offset + numel].view(p.size()).clone()
                offset += numel

        # Per-mixer gradient clipping
        if self.config.max_grad_norm > 0:
            for i in range(n_outputs):
                torch.nn.utils.clip_grad_norm_(
                    self._opt_params_list[i], self.config.max_grad_norm
                )

        return avg_loss

    def _outer_step_adam(self, tasks: List[PDETask]) -> float:
        """Adam outer step: compute gradient, then step each mixer's optimizer."""
        avg_loss = self._compute_meta_gradient(tasks)
        for opt_i in self.outer_opts:
            opt_i.step()  # type: ignore[call-arg]
        return avg_loss

    def _outer_step_lbfgs(self, tasks: List[PDETask]) -> float:
        """L-BFGS outer step: per-mixer sequential line searches.

        Each mixer's L-BFGS runs its own line search against the same
        task batch. The closure recomputes the full meta-gradient at
        each probe; L-BFGS reads only its own parameter slice's grads
        so other mixers' gradients are ignored within one optimizer's
        step. Mixers run sequentially, so mixer_{i+1} sees parameters
        already moved by mixer_i's step.
        """
        result = 0.0
        for i, opt_i in enumerate(self.outer_opts):
            probe = 0

            def closure(mixer_idx: int = i) -> float:
                nonlocal result, probe
                probe += 1
                print(f"\t[L-BFGS outer mixer {mixer_idx} probe {probe}]")
                self.model.zero_grad()
                result = self._compute_meta_gradient(tasks)
                return result

            opt_i.step(closure)  # type: ignore[arg-type]
        return result

    def outer_step(self, tasks: List[PDETask]) -> float:
        """
        Perform one iMAML meta-update step.

        For Adam: compute gradient, then Adam.step() per mixer.
        For L-BFGS: per-mixer sequential line searches, closure recomputes
        gradient at trial points. Same task batch used across all closure
        calls within one .step().

        Returns:
            Average query loss across tasks
        """
        self.model.zero_grad()

        avg_loss = self._outer_step(tasks)

        # Per-mixer lambda update — each mixer's lambda is independent.
        if self.lam_lr > 0:
            for i in range(len(self.lam)):
                lam_delta = -self.lam_lr * self._lam_grad[i]
                self.lam[i] = torch.clamp(
                    self.lam[i] + lam_delta, self.lam_min, 5000.0
                )

        return avg_loss

    def evaluate(self, tasks: List[PDETask], seed: int = 0) -> float:
        """
        Evaluate meta-learned initialization on tasks (no meta-update).

        Args:
            tasks: Tasks to evaluate on
            seed: Base seed for support/query splits

        Returns:
            Average query loss after adaptation
        """
        self.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            # Note: We still need gradients for inner loop, but not for outer
            # So we use torch.enable_grad() inside compute_task_loss
            pass

        for i, task in enumerate(tasks):
            task_seed = seed + i
            with torch.enable_grad():
                _, _, task_loss_val, _ = self.compute_task(task, task_seed)
            total_loss += task_loss_val

        return total_loss / len(tasks)

    # ------------------------------------------------------------------
    # Iteration hooks: bound at __init__ based on checkpoint mode
    # ------------------------------------------------------------------

    def _patience_iteration_hook(
        self, iteration: int, train_loss: float, log_interval: int, checkpoint_dir: Path
    ) -> bool:
        """Per-iteration logic for patience mode: best-tracking + validation."""
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            self.best_train_state = copy.deepcopy(self.model.state_dict())
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        lrs = [f"{opt_i.param_groups[0]['lr']:.2e}" for opt_i in self.outer_opts]
        lr_str = (
            f", lr=[{', '.join(lrs)}]"
            if any(s is not None for s in self.schedulers)
            else ""
        )
        lam_str = ",".join(f"{l.item():.4f}" for l in self.lam)
        print(
            f"Iter {iteration + 1:5d}: train_loss={train_loss:.6f}, patience={self.patience_counter}{lr_str}, λ=[{lam_str}]"
        )

        if (
            self.patience_counter >= self.config.patience
            and self.val_loader is not None
        ):
            if self.validate(train_loss, checkpoint_dir):
                self.patience_counter = 0
            else:
                return True  # early_stopped
        return False

    def _interval_iteration_hook(
        self, iteration: int, train_loss: float, log_interval: int, checkpoint_dir: Path
    ) -> bool:
        """Per-iteration logic for interval mode: periodic saves, no best-tracking."""
        lrs = [f"{opt_i.param_groups[0]['lr']:.2e}" for opt_i in self.outer_opts]
        lr_str = (
            f", lr=[{', '.join(lrs)}]"
            if any(s is not None for s in self.schedulers)
            else ""
        )
        lam_str = ",".join(f"{l.item():.4f}" for l in self.lam)
        print(f"Iter {iteration + 1:5d}: train_loss={train_loss:.6f}{lr_str}, λ=[{lam_str}]")

        if (
            iteration > 0
            and (iteration % self.config.checkpoint_interval == 0)
        ):
            self.save_checkpoint(checkpoint_dir / "latest_model.pt")
            print(f"  Checkpoint saved at iteration {iteration}.")
        return False

    def _patience_finalize(
        self, train_loss: float, early_stopped: bool, checkpoint_dir: Path
    ) -> None:
        """End-of-training for patience mode."""
        if self.val_loader is not None and not early_stopped:
            print("Doing end-of-loop validation run.")
            self.validate(train_loss, checkpoint_dir)
        else:
            if self.best_train_state is not None:
                self.model.load_state_dict(self.best_train_state)
            if not (checkpoint_dir / "final_model.pt").exists():
                print("Saving final model (no validation checkpoint was saved)...")
                self.save_checkpoint(checkpoint_dir / "final_model.pt")
        print(f"  Best train loss: {self.best_train_loss:.6f}")
        if self.val_loader is not None:
            print(f"  Best val loss: {self.best_val_loss:.6f}")

    def _interval_finalize(
        self, train_loss: float, early_stopped: bool, checkpoint_dir: Path
    ) -> None:
        """End-of-training for interval mode: save final weights."""
        self.save_checkpoint(checkpoint_dir / "final_model.pt")
        print(f"Saved final model to {checkpoint_dir / 'final_model.pt'}.")
        print(f"  Final train loss: {train_loss:.6f}")

    def validate(
        self,
        train_loss: float,
        checkpoint_dir: Path,
    ) -> bool:
        print(
            "  → Patience exhausted. Comparing current vs best-train on validation..."
        )

        assert self.val_loader is not None
        assert self.best_train_state is not None

        # Stash current state before we modify model
        current_state = copy.deepcopy(self.model.state_dict())

        # Evaluate current weights on validation
        current_val_loss = self.evaluate(self.val_loader.tasks)
        print(f"  → Current weights val_loss: {current_val_loss:.6f}")

        # Evaluate best-train weights on validation
        self.model.load_state_dict(self.best_train_state)
        best_train_val_loss = self.evaluate(self.val_loader.tasks)
        winner_train_loss = None
        print(f"  → Best-train weights val_loss: {best_train_val_loss:.6f}")

        # Pick winner
        if current_val_loss < best_train_val_loss:
            winner_state = current_state
            winner_val_loss = current_val_loss
            winner_train_loss = train_loss
            print("  → Winner: current weights")
        else:
            winner_state = self.best_train_state
            winner_val_loss = best_train_val_loss
            winner_train_loss = self.best_train_loss
            print("  → Winner: best-train weights")

        self.history["val_loss"].append(winner_val_loss)

        # Did winner beat previous best val?
        if winner_val_loss < self.best_val_loss:
            self.best_val_loss = winner_val_loss
            print("  → Validation improved! Resetting patience.")

            print(
                f"  → Saved checkpoint to {checkpoint_dir / 'final_model.pt'}. Train loss: {winner_train_loss:.6f}, Val loss: {winner_val_loss:.6f}"
            )
            print(f"      - Current weights train_loss: {train_loss}")
            print(f"      - Best-train weights train_loss: {self.best_train_loss}")

            # Continue training from winner weights
            self.model.load_state_dict(winner_state)
            self.best_train_state = copy.deepcopy(winner_state)
            self.best_train_loss = winner_train_loss  # Reset train baseline

            self.save_checkpoint(checkpoint_dir / "final_model.pt")
            self.save_checkpoint(checkpoint_dir / "latest_model.pt")
            return True
        else:
            print("  → Validation stalled. Early stopping.")
            self.save_checkpoint(checkpoint_dir / "latest_model.pt")
            return False

    def train(
        self, checkpoint_dir: Optional[Path] = None, log_interval: Optional[int] = None
    ) -> tuple[Dict[str, List], bool]:
        """
        Run full MAML training loop with two-level early stopping.

        Two-level early stopping:
        1. Track meta-train loss improvement (patience counter)
        2. When patience exhausted, check validation loss
        3. If validation improved, reset patience and save checkpoint
        4. If validation also stalled, stop training

        Args:
            checkpoint_dir: Directory to save checkpoints (None = no saving)
            log_interval: Print loss every N iterations (None = use config)

        Returns:
            Training history dict with 'train_loss', 'val_loss', 'iteration'
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("checkpoint_dir must be specified for saving checkpoints.")

        if log_interval is None:
            log_interval = self.config.log_interval

        # On fresh start, reset loop state. On resume, already loaded from checkpoint.
        if not self._resumed:
            self.patience_counter = 0
            self.history = self._fresh_history()

        start_iteration = (self.iteration + 1) if self._resumed else 0

        print("Starting iMAML training:")
        print(f"  Device: {self.device}")
        print(f"  Inner LR (α): {self.config.inner_lr}")
        print(f"  Outer LR (β): {self.config.outer_lr}")
        print(f"  Inner steps: {self.config.inner_steps}")
        print(f"  Meta batch size: {self.config.meta_batch_size}")
        print(f"  K-shot: {self.config.k_shot}")
        print(f"  Query size: {self.config.query_size}")
        lam_init = ",".join(f"{l.item():.4f}" for l in self.lam)
        print(f"  λ (proximal): [{lam_init}]")
        print(f"  CG steps: {self.cg_steps}")
        print(f"  CG damping: {self.cg_damping}")
        print(f"  Inner optimizer: {self.config.imaml.inner_optimizer}")
        if self.lam_lr > 0:
            print(f"  λ meta-learning: lr={self.lam_lr}")
        if self.config.warmup_iterations > 0:
            print(f"  Warmup: {self.config.warmup_iterations} iterations")
        if self.config.use_scheduler:
            sched_desc = f"cosine annealing to {self.config.min_lr}"
            if self.config.scheduler_type == "warm_restarts":
                sched_desc += f" (warm restarts: T_0={self.config.T_0}, T_mult={self.config.T_mult})"
            print(f"  LR scheduler: {sched_desc}")
        if self.config.max_grad_norm > 0:
            print(f"  Grad clip (inner+outer): max_norm={self.config.max_grad_norm}")
        if self._patience_enabled:
            print(f"  Checkpoint mode: patience={self.config.patience}")
        else:
            print(f"  Checkpoint mode: interval={self.config.checkpoint_interval}")
        if self._resumed:
            print(f"  Resuming from iteration {start_iteration}")
        print()

        self._stop_requested = False

        # Catch Ctrl+C as a stop request — save happens at the iteration boundary
        prev_handler = signal.signal(
            signal.SIGINT, lambda *_: setattr(self, "_stop_requested", True)
        )

        im = self.config.imaml
        if im.outer_optimizer == "adam+lbfgs":
            # Phase 1: Adam
            switch_at = im.outer_lbfgs_after
            phase1_start = max(start_iteration, 0)
            if phase1_start < switch_at:
                print(f"  Phase 1: Adam outer, iterations {phase1_start}→{switch_at}")
                result = self._run_phase(phase1_start, switch_at, checkpoint_dir, log_interval)
                if result is not None:
                    signal.signal(signal.SIGINT, prev_handler)
                    return result, False

            # Phase 2: switch to L-BFGS outer
            phase2_start = max(start_iteration, switch_at)
            if phase2_start < self.config.max_iterations:
                print(f"  Phase 2: L-BFGS outer (full batch), iterations {phase2_start}→{self.config.max_iterations}")
                self.outer_opts = [
                    torch.optim.LBFGS(
                        params_i, lr=1.0, max_iter=1, max_eval=20,
                        tolerance_grad=1e-15, tolerance_change=1e-15,
                        line_search_fn="strong_wolfe",
                    )
                    for params_i in self._opt_params_list
                ]
                self._outer_step = self._outer_step_lbfgs
                self.schedulers = [None] * len(self.outer_opts)  # L-BFGS doesn't use scheduler
                # Force full batch for L-BFGS (deterministic objective for line search)
                self.config.meta_batch_size = len(self.train_loader.tasks)
                result = self._run_phase(phase2_start, self.config.max_iterations, checkpoint_dir, log_interval)
                if result is not None:
                    signal.signal(signal.SIGINT, prev_handler)
                    return result, False
        elif self.config.da_enabled:
            # DA for iMAML: CG=0 (FOMAML) until da_threshold, then CG=config value
            da_end = min(self.config.da_threshold, self.config.max_iterations)
            phase1_start = max(start_iteration, 0)
            if phase1_start < da_end:
                print(f"  DA Phase 1: CG=0 (FOMAML), iterations {phase1_start}→{da_end}")
                self._cg_steps_active = 0
                result = self._run_phase(phase1_start, da_end, checkpoint_dir, log_interval)
                if result is not None:
                    signal.signal(signal.SIGINT, prev_handler)
                    return result, False

            phase2_start = max(start_iteration, da_end)
            if phase2_start < self.config.max_iterations:
                print(f"  DA Phase 2: CG={self.config.imaml.cg_steps}, iterations {phase2_start}→{self.config.max_iterations}")
                self._cg_steps_active = self.config.imaml.cg_steps
                result = self._run_phase(phase2_start, self.config.max_iterations, checkpoint_dir, log_interval)
                if result is not None:
                    signal.signal(signal.SIGINT, prev_handler)
                    return result, False
        else:
            per_epoch = self.config.max_iterations
            total = per_epoch * self.config.epochs
            start_epoch = start_iteration // per_epoch
            for epoch in range(start_epoch, self.config.epochs):
                epoch_start = max(start_iteration, epoch * per_epoch)
                epoch_end = (epoch + 1) * per_epoch
                if epoch > 0 and (epoch > start_epoch or start_iteration == epoch * per_epoch):
                    # Reset scheduler + optimizer for new epoch (keep model weights)
                    self._reset_for_epoch()
                    print(f"\n  Epoch {epoch + 1}/{self.config.epochs}: reset scheduler + optimizer")
                result = self._run_phase(epoch_start, epoch_end, checkpoint_dir, log_interval)
                if result is not None:
                    signal.signal(signal.SIGINT, prev_handler)
                    return result, False

        # Restore original handler
        signal.signal(signal.SIGINT, prev_handler)

        self._training_finalize(self._last_train_loss, self._early_stopped, checkpoint_dir)

        print()
        print(f"Training complete at iteration {self.iteration + 1}")

        return self.history, True

    def _run_phase(
        self,
        start_iter: int,
        end_iter: int,
        checkpoint_dir: Path,
        log_interval: int,
    ) -> Optional[Dict[str, List]]:
        """Run training iterations from start_iter to end_iter.

        Returns None on normal completion (including early stop via patience).
        Returns self.history on NaN or Ctrl+C (caller should return immediately).
        """
        for iteration in range(start_iter, end_iter):
            self.iteration = iteration

            tasks = self.train_loader.sample_batch(
                self.config.meta_batch_size, seed=(iteration % self.config.max_iterations)
            )

            # Suppress per-task prints to stdout (still logged to buffer)
            # Always print first iteration of each phase (start or resume)
            verbose = (iteration == start_iter) or (iteration % log_interval == 0)
            if hasattr(sys.stdout, 'quiet'):
                sys.stdout.quiet = not verbose  # type: ignore[attr-defined]

            train_loss = self.outer_step(tasks)

            if hasattr(sys.stdout, 'quiet'):
                sys.stdout.quiet = False  # type: ignore[attr-defined]
            self._last_train_loss = train_loss

            if torch.tensor(train_loss).isnan():
                self._nan_at = iteration
                print(f"\n  NaN detected at iteration {iteration}. Weights are clean (step was skipped).")

                if self._patience_enabled:
                    if self.val_loader is not None and self.best_train_state is not None:
                        current_state = copy.deepcopy(self.model.state_dict())
                        current_val = self.evaluate(self.val_loader.tasks)
                        print(f"  Current weights val_loss: {current_val:.6f}")

                        self.model.load_state_dict(self.best_train_state)
                        best_val = self.evaluate(self.val_loader.tasks)
                        print(f"  Best-train weights val_loss: {best_val:.6f}")

                        if current_val <= best_val:
                            self.model.load_state_dict(current_state)
                            print("  Winner: current weights")
                        else:
                            print("  Winner: best-train weights")
                    elif self.best_train_state is not None:
                        self.model.load_state_dict(self.best_train_state)
                        print("  Restored best-train weights (no val loader).")

                self.save_checkpoint(checkpoint_dir / "final_model.pt")
                print(f"  Saved to {checkpoint_dir / 'final_model.pt'}.")
                return self.history

            if self.config.has_scheduler:
                if self._plateau_mode:
                    # Feed rolling average to ReduceLROnPlateau
                    self._loss_window.append(train_loss)
                    if len(self._loss_window) > self.config.plateau_window:
                        self._loss_window.pop(0)
                    rolling_avg = sum(self._loss_window) / len(self._loss_window)
                    for sched_i in self.schedulers:
                        if sched_i is not None:
                            sched_i.step(rolling_avg)  # type: ignore[call-arg]
                else:
                    for sched_i in self.schedulers:
                        if sched_i is not None:
                            sched_i.step()

            self.history["train_loss"].append(train_loss)
            self.history["iteration"].append(iteration)

            # M5a expanded logging — append the per-iter metrics blob built
            # by _compute_meta_gradient. Only appended after the NaN check
            # so history lists stay length-aligned with train_loss/iteration.
            if self._last_iter_metrics is not None:
                self.history["lam"].append(self._last_iter_metrics["lam"])
                self.history["lr"].append(self._last_iter_metrics["lr"])
                for mixer_key, mixer_blob in self._last_iter_metrics["mixers"].items():
                    h_mixer = self.history["mixers"][mixer_key]
                    h_mixer["support_pre_adapt"].append(mixer_blob["support_pre_adapt"])
                    h_mixer["support_post_adapt"].append(mixer_blob["support_post_adapt"])
                    h_mixer["mse_main"].append(mixer_blob["mse_main"])
                    for name, v in mixer_blob["aux"].items():
                        h_mixer["aux"][name].append(v)
                    if self.aux_losses_enabled:
                        h_mixer["s_mse"].append(mixer_blob["s_mse"])
                        h_mixer["eff_weight_mse"].append(mixer_blob["eff_weight_mse"])
                        for name, v in mixer_blob["s_aux"].items():
                            h_mixer["s_aux"][name].append(v)
                        for name, v in mixer_blob["eff_weight_aux"].items():
                            h_mixer["eff_weight_aux"][name].append(v)

            early_stopped = self._iteration_hook(iteration, train_loss, log_interval, checkpoint_dir)
            if early_stopped:
                self._early_stopped = True
                return None

            if self._stop_requested:
                print(
                    f"\n  Stop requested after iteration {iteration}. Saving checkpoint..."
                )
                self.save_checkpoint(checkpoint_dir / "latest_model.pt")
                print(
                    f"  Saved to {checkpoint_dir / 'latest_model.pt'}. Resume with --resume."
                )
                return self.history

        return None

    def pop_nan_iteration(self) -> Optional[int]:
        """Return the iteration where NaN was detected, or None. Resets the flag."""
        it = self._nan_at
        self._nan_at = None
        return it

    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dicts": [opt_i.state_dict() for opt_i in self.outer_opts],
                "config": self.config,
                "iteration": self.iteration,
                "best_train_loss": self.best_train_loss,
                "best_val_loss": self.best_val_loss,
                "best_train_state": self.best_train_state,
                "scheduler_state_dicts": [
                    sched_i.state_dict() if sched_i is not None else None
                    for sched_i in self.schedulers
                ],
                "rng_state_cpu": torch.random.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else None,
                "patience_counter": self.patience_counter,
                "history": self.history,
                "lam": self.lam,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Hard check: aux_losses_enabled must match between the checkpoint
        # and the current config. The model structure (Kendall log-variances
        # registered or not) depends on this flag, so mismatched loads would
        # either crash in load_state_dict or silently skip parameters.
        ckpt_cfg = checkpoint.get("config")
        if ckpt_cfg is not None:
            ckpt_aux = ckpt_cfg.aux_losses_enabled
            if ckpt_aux != self.aux_losses_enabled:
                raise RuntimeError(
                    f"aux_losses_enabled mismatch: checkpoint was trained "
                    f"with aux_losses_enabled={ckpt_aux}, current config "
                    f"has {self.aux_losses_enabled}. Model structure differs "
                    f"(Kendall log-variances registered vs not) so the "
                    f"checkpoint cannot be loaded into this trainer."
                )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        opt_states = checkpoint["optimizer_state_dicts"]
        if len(opt_states) != len(self.outer_opts):
            raise RuntimeError(
                f"optimizer count mismatch: checkpoint has {len(opt_states)} "
                f"optimizers, current trainer has {len(self.outer_opts)}. "
                f"This usually means n_outputs differs between checkpoint "
                f"and current config."
            )
        for opt_i, state_i in zip(self.outer_opts, opt_states):
            opt_i.load_state_dict(state_i)
        self.iteration = checkpoint.get("iteration", 0)
        self.best_train_loss = checkpoint.get("best_train_loss", float("inf"))
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_train_state = checkpoint.get("best_train_state", None)

        # Restore scheduler states
        sched_states = checkpoint.get("scheduler_state_dicts")
        if sched_states is not None:
            for sched_i, state_i in zip(self.schedulers, sched_states):
                if sched_i is not None and state_i is not None:
                    sched_i.load_state_dict(state_i)

        # Restore RNG states for exact replay
        rng_cpu = checkpoint.get("rng_state_cpu")
        if rng_cpu is not None:
            torch.random.set_rng_state(rng_cpu.cpu().to(torch.uint8))
        rng_cuda = checkpoint.get("rng_state_cuda")
        if rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_cuda.cpu().to(torch.uint8))

        # Restore training loop state
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.history = checkpoint.get(
            "history", {"train_loss": [], "val_loss": [], "iteration": []}
        )
        self._resumed = True

        # Restore lambda (may have been meta-learned per-mixer)
        saved_lam = checkpoint.get("lam")
        if saved_lam is not None:
            if isinstance(saved_lam, list):
                self.lam = [l.to(self.device) for l in saved_lam]
            else:
                # Legacy single-tensor checkpoint — broadcast to all mixers
                self.lam = [saved_lam.to(self.device) for _ in range(len(self.lam))]
            lam_str = ",".join(f"{l.item():.4f}" for l in self.lam)
            print(f"  Loaded λ=[{lam_str}]")

        print(f"Loaded checkpoint from {path}")
        print(f"  Iteration: {self.iteration}")
        if self._patience_enabled:
            print(f"  Best train loss: {self.best_train_loss:.6f}")
            print(f"  Best val loss: {self.best_val_loss:.6f}")


def get_meta_learned_init(
    checkpoint_path: Path, model_class: type, **model_kwargs
) -> nn.Module:
    """
    Load meta-learned initialization θ* from checkpoint.

    Args:
        checkpoint_path: Path to MAML checkpoint
        model_class: Model class to instantiate (default: PDEOperatorNetwork)
        **model_kwargs: Arguments passed to model constructor

    Returns:
        Model with meta-learned weights loaded
    """

    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def fine_tune(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    lr: float,
    max_steps: int,
    device: str = "cpu",
) -> List[float]:
    """
    Fine-tune model on given data and return loss curve.

    Args:
        model: Model to fine-tune (will be modified in-place)
        features: Input features tensor (N, input_dim) on device
        targets: Target outputs tensor (N, output_dim) on device
        lr: Learning rate for fine-tuning
        max_steps: Number of gradient steps
        device: Device to run on

    Returns:
        List of loss values at each step
    """
    model = model.to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    x = features
    y = targets

    losses = []
    for _ in range(max_steps):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    return losses
