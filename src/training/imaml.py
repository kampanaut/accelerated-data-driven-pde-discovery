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
from typing import Callable, List, Optional, Dict, Tuple, TYPE_CHECKING

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
        elif loss_type == "mae":
            self._pointwise_loss = F.l1_loss
        else:
            raise ValueError(
                f"Unknown loss_function: {loss_type}. Use 'mse', 'normalized_mse', or 'mae'."
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
        self.lam = torch.tensor(im.lam, device=self.device, dtype=torch.float32)
        self.lam_lr = im.lam_lr
        self.lam_min = im.lam_min
        self.cg_steps = im.cg_steps
        self.cg_damping = im.cg_damping

        # Bind inner solver — SGD (prox every step or at end) or L-BFGS
        if im.inner_optimizer == "lbfgs":
            self._inner_solve = self._inner_solve_lbfgs
        elif im.proximal_every_step:
            self._inner_solve = self._inner_solve_sgd
        else:
            self._inner_solve = self._inner_solve_sgd_prox_end
        prox_mode = "every_step" if im.proximal_every_step else "end_only"
        print(f"  iMAML: lam={im.lam}, cg_steps={im.cg_steps}, "
              f"cg_damping={im.cg_damping}, inner_optimizer={im.inner_optimizer}, "
              f"proximal={prox_mode}")

        # Outer loop optimizer (meta-update) — model params only
        opt_params = list(self.model.parameters())
        self.outer_opt = torch.optim.Adam(opt_params, lr=t.outer_lr, betas=tuple(t.adam_betas))

        # LR scheduler: optional warmup → cosine decay (single or warm restarts)
        self.scheduler = None
        if t.use_scheduler or t.warmup_iterations > 0:
            schedulers = []
            milestones = []

            # Phase 1: linear warmup
            if t.warmup_iterations > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.outer_opt,
                    start_factor=1.0 / t.warmup_iterations,
                    end_factor=1.0,
                    total_iters=t.warmup_iterations,
                )
                schedulers.append(warmup)
                milestones.append(t.warmup_iterations)

            # Phase 2: cosine decay
            if t.use_scheduler:
                post_warmup_iters = (
                    t.max_iterations - t.warmup_iterations
                )
                if t.scheduler_type == "warm_restarts":
                    decay = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.outer_opt,
                        T_0=t.T_0,
                        T_mult=t.T_mult,
                        eta_min=t.min_lr,
                    )
                else:
                    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.outer_opt,
                        T_max=post_warmup_iters,
                        eta_min=t.min_lr,
                    )
                schedulers.append(decay)

            # Chain them
            if len(schedulers) > 1:
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.outer_opt,
                    schedulers=schedulers,
                    milestones=milestones,
                )
            elif schedulers:
                self.scheduler = schedulers[0]

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

        # Mutual exclusion: iMAML is incompatible with MAML-specific features
        for flag, name in [(t.msl_enabled, "MSL"), (t.da_enabled, "DA"),
                           (t.lslr_enabled, "LSLR"), (t.metal.enabled, "MeTAL")]:
            if flag:
                raise ValueError(f"{name} is MAML-specific, incompatible with iMAML.")

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
        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "iteration": [],
        }
        self._resumed = False
        self._stop_requested = False

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

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        """Return flattened parameter vector (detached clone)."""
        return torch.cat([p.data.view(-1) for p in model.parameters()]).clone()

    def _set_flat_params(self, model: nn.Module, flat: torch.Tensor) -> None:
        """Write flattened vector back into model parameters."""
        offset = 0
        for p in model.parameters():
            p.data.copy_(flat[offset:offset + p.nelement()].view(p.size()))
            offset += p.nelement()

    def _regularization_loss(
        self, model: nn.Module, theta: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """Proximal term: λ/2 ||φ - θ||² (Eq. 3 in Rajeswaran et al. 2019)."""
        phi = torch.cat([p.view(-1) for p in model.parameters()])
        return 0.5 * lam * (phi - theta).pow(2).sum()

    # ------------------------------------------------------------------
    # Inner solvers (bound in __init__)
    # ------------------------------------------------------------------

    def _inner_solve_sgd(
        self,
        fast_model: nn.Module,
        theta: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Inner loop: SGD on task_loss + proximal for inner_steps steps (Eq. 3)."""
        opt = torch.optim.SGD(fast_model.parameters(), lr=self.config.inner_lr)
        for _ in range(self.config.inner_steps):
            opt.zero_grad()
            pred = fast_model(support_x)
            task_loss = self.cost_function(pred, support_y, support_coords)
            prox = self._regularization_loss(fast_model, theta, self.lam)
            total = task_loss + prox
            total.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(fast_model.parameters(), self.config.max_grad_norm)
            opt.step()

    def _inner_solve_sgd_prox_end(
        self,
        fast_model: nn.Module,
        theta: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Inner loop: plain SGD then one proximal step at end (reference code approach)."""
        opt = torch.optim.SGD(fast_model.parameters(), lr=self.config.inner_lr)
        for _ in range(self.config.inner_steps):
            opt.zero_grad()
            pred = fast_model(support_x)
            task_loss = self.cost_function(pred, support_y, support_coords)
            task_loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(fast_model.parameters(), self.config.max_grad_norm)
            opt.step()
        # One proximal pull-back step
        opt.zero_grad()
        prox = self._regularization_loss(fast_model, theta, self.lam)
        prox.backward()
        opt.step()

    def _inner_solve_lbfgs(
        self,
        fast_model: nn.Module,
        theta: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Inner loop: L-BFGS on task_loss + proximal (Raissi 2018 style)."""
        opt = torch.optim.LBFGS(
            fast_model.parameters(),
            lr=1.0,
            max_iter=self.config.inner_steps,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            pred = fast_model(support_x)
            task_loss = self.cost_function(pred, support_y, support_coords)
            prox = self._regularization_loss(fast_model, theta, self.lam)
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
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute H·v where H = ∇²φ L̂(φ) at current model params.

        Uses double autograd.grad trick (Rajeswaran et al. 2019).
        The Hessian is of the task loss only — the proximal term's Hessian
        is λI, already accounted for in the matrix evaluator.
        """
        pred = model(support_x)
        task_loss = self.cost_function(pred, support_y, support_coords)
        grad_ft = torch.autograd.grad(task_loss, model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        h = torch.sum(flat_grad * vector)
        hvp = torch.autograd.grad(h, model.parameters())
        return torch.cat([g.contiguous().view(-1) for g in hvp])

    def _matrix_evaluator(
        self,
        model: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build linear operator A for CG: Av = v + Hv / (lam + damping).

        Corresponds to (I + 1/λ H) from Lemma 1 (Eq. 6), with damping
        added to λ for numerical stability.
        """
        lam = self.lam
        damping = self.cg_damping

        def evaluator(v: torch.Tensor) -> torch.Tensor:
            hvp = self._hessian_vector_product(
                model, support_x, support_y, support_coords, v
            )
            return v + hvp / (lam + damping)

        return evaluator

    # ------------------------------------------------------------------
    # Core iMAML task computation
    # ------------------------------------------------------------------

    def _task_setup(
        self, task: PDETask, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[Tuple[torch.Tensor, torch.Tensor]],
               Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Common setup: split, zero features, set domain."""
        support, query, support_coords, query_coords = task.get_support_query_split(
            K_shot=self.config.k_shot,
            query_size=self.config.query_size,
            k_seed=seed,
        )

        if self.config.inner_steps == 0:
            raise ValueError("You cannot run training with inner_steps == 0")

        support_x, support_y = support
        query_x, query_y = query

        if self.config.zero_non_rhs_features:
            support_x = task.zero_non_rhs_features(support_x)
            query_x = task.zero_non_rhs_features(query_x)

        self._current_Lx = task.Lx
        self._current_Ly = task.Ly

        return support_x, support_y, query_x, query_y, support_coords, query_coords

    def _compute_task_imaml(
        self, task: PDETask, seed: int
    ) -> Tuple[torch.Tensor, float]:
        """One task: copy θ → inner solve → query grad → CG correct.

        Returns (corrected_grad, query_loss_value). Unlike MAML, the meta-gradient
        is not computed via backprop — it's a flat vector from CG.
        """
        support_x, support_y, query_x, query_y, support_coords, query_coords = (
            self._task_setup(task, seed)
        )

        theta = self._get_flat_params(self.model)
        fast_model = copy.deepcopy(self.model)

        verbose = not getattr(sys.stdout, 'quiet', False)

        if verbose:
            with torch.no_grad():
                pre_pred = fast_model(support_x)
                pre_loss = self.cost_function(pre_pred, support_y, support_coords)
                print(f"\t\tpre-adapt: support_loss={pre_loss.item():.6f}")

        # Inner solve: minimize task_loss + λ/2 ||φ - θ||²
        self._inner_solve(fast_model, theta, support_x, support_y, support_coords)

        if verbose:
            with torch.no_grad():
                post_pred = fast_model(support_x)
                post_loss = self.cost_function(post_pred, support_y, support_coords)
                print(f"\t\tpost-adapt: support_loss={post_loss.item():.6f}"
                      f" ({self.config.inner_steps} steps)")

        # Query gradient: v_i = ∇φ L_query(φ*)
        query_pred = fast_model(query_x)
        query_loss = self.cost_function(query_pred, query_y, query_coords)
        query_grad = torch.autograd.grad(query_loss, fast_model.parameters())
        flat_grad = torch.cat([g.contiguous().view(-1) for g in query_grad])
        query_loss_val = query_loss.item()

        # CG solve: g_i = (I + 1/λ H)⁻¹ v_i
        if self.cg_steps <= 1:
            corrected_grad = flat_grad.detach()
        else:
            evaluator = self._matrix_evaluator(
                fast_model, support_x, support_y, support_coords
            )
            corrected_grad = cg_solve(evaluator, flat_grad.detach(), self.cg_steps)

        if verbose:
            print(f"\t\t\tloss={query_loss_val:.2f}")

        return corrected_grad, query_loss_val

    def outer_step(self, tasks: List[PDETask]) -> float:
        """
        Perform one iMAML meta-update step.

        Accumulates CG-corrected gradient vectors (not loss tensors),
        then sets .grad on model params and calls outer_opt.step().

        Returns:
            Average query loss across tasks
        """
        self.model.train()
        self.outer_opt.zero_grad()

        n_params = sum(p.numel() for p in self.model.parameters())
        meta_grad = torch.zeros(n_params, device=self.device)
        total_loss = 0.0
        lam_grad = 0.0

        for i, task in enumerate(tasks):
            print(f"\ttask [{i}/{len(tasks)}]: {task.task_name}")
            seed = self.iteration * len(tasks) + i
            corrected_grad, loss_val = self.compute_task(task, seed)
            meta_grad += corrected_grad / len(tasks)
            total_loss += loss_val / len(tasks)

            # Optional lambda meta-learning
            if self.lam_lr > 0:
                # Compute support gradient at adapted point for lambda update
                # (already available from CG — recompute cheaply)
                fast_model = copy.deepcopy(self.model)
                theta = self._get_flat_params(self.model)
                support_x, support_y, _, _, support_coords, _ = self._task_setup(task, seed)
                self._inner_solve(fast_model, theta, support_x, support_y, support_coords)
                train_loss = self.cost_function(fast_model(support_x), support_y, support_coords)
                train_grad_t = torch.autograd.grad(train_loss, fast_model.parameters())
                flat_train_grad = torch.cat([g.view(-1) for g in train_grad_t]).detach()
                inner_prod = flat_train_grad.dot(corrected_grad)
                task_lam_grad = inner_prod / (self.lam ** 2 + 0.1)
                lam_grad += task_lam_grad / len(tasks)

        # Check for NaN
        avg_loss = total_loss
        if not torch.tensor(avg_loss).isfinite():
            return float("nan")

        # Set .grad on model params from flat meta_grad vector
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.grad = meta_grad[offset:offset + numel].view(p.size()).clone()
            offset += numel

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.outer_opt.step()

        # Lambda update
        if self.lam_lr > 0:
            lam_delta = -self.lam_lr * lam_grad
            self.lam = torch.clamp(self.lam + lam_delta, self.lam_min, 5000.0)

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
                _, task_loss_val = self.compute_task(task, task_seed)
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

        lr_str = (
            f", lr={self.outer_opt.param_groups[0]['lr']:.2e}"
            if self.scheduler
            else ""
        )
        print(
            f"Iter {iteration + 1:5d}: train_loss={train_loss:.6f}, patience={self.patience_counter}{lr_str}"
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
        lr_str = (
            f", lr={self.outer_opt.param_groups[0]['lr']:.2e}"
            if self.scheduler
            else ""
        )
        print(f"Iter {iteration + 1:5d}: train_loss={train_loss:.6f}{lr_str}")

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
            self.history = {"train_loss": [], "val_loss": [], "iteration": []}

        start_iteration = (self.iteration + 1) if self._resumed else 0

        print("Starting iMAML training:")
        print(f"  Device: {self.device}")
        print(f"  Inner LR (α): {self.config.inner_lr}")
        print(f"  Outer LR (β): {self.config.outer_lr}")
        print(f"  Inner steps: {self.config.inner_steps}")
        print(f"  Meta batch size: {self.config.meta_batch_size}")
        print(f"  K-shot: {self.config.k_shot}")
        print(f"  Query size: {self.config.query_size}")
        print(f"  λ (proximal): {self.lam.item():.4f}")
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

        result = self._run_phase(start_iteration, self.config.max_iterations, checkpoint_dir, log_interval)
        if result is not None:
            signal.signal(signal.SIGINT, prev_handler)
            return result, False

        # Restore original handler
        signal.signal(signal.SIGINT, prev_handler)

        self._training_finalize(self._last_train_loss, self._early_stopped, checkpoint_dir)

        print()
        print(f"Training complete at iteration {self.iteration}")

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
                self.config.meta_batch_size, seed=iteration
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

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["iteration"].append(iteration)

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
                "optimizer_state_dict": self.outer_opt.state_dict(),
                "config": self.config,
                "iteration": self.iteration,
                "best_train_loss": self.best_train_loss,
                "best_val_loss": self.best_val_loss,
                "best_train_state": self.best_train_state,
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
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

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.outer_opt.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint.get("iteration", 0)
        self.best_train_loss = checkpoint.get("best_train_loss", float("inf"))
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_train_state = checkpoint.get("best_train_state", None)

        # Restore scheduler state
        sched_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and sched_state is not None:
            self.scheduler.load_state_dict(sched_state)

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

        # Restore lambda (may have been meta-learned)
        saved_lam = checkpoint.get("lam")
        if saved_lam is not None:
            self.lam = saved_lam.to(self.device)
            print(f"  Loaded λ={self.lam.item():.4f}")

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
