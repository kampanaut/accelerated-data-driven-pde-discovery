"""
MAML (Model-Agnostic Meta-Learning) implementation for PDE discovery.

Uses the `higher` library for automatic functional parameter handling,
enabling gradient computation through inner loop optimization steps.

References:
- Finn et al. 2017: "Model-Agnostic Meta-Learning for Fast Adaptation"
- Nichol et al. 2018: "On First-Order Meta-Learning Algorithms" (FOMAML)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Dict, Tuple
import copy
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from .task_loader import PDETask, MetaLearningDataLoader
from .spectral_loss import compute_spectral_loss


# ---------------------------------------------------------------------------
# Configurable loss function
# ---------------------------------------------------------------------------


def compute_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_type: str
) -> torch.Tensor:
    """
    Compute loss between prediction and target.

    Args:
        pred: Model predictions (N, output_dim)
        target: Ground truth (N, output_dim)
        loss_type: One of 'mse', 'normalized_mse', 'mae'

    Returns:
        Scalar loss tensor
    """
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    elif loss_type == "normalized_mse":
        return F.mse_loss(pred, target) / (target**2).mean()
    elif loss_type == "mae":
        return F.l1_loss(pred, target)
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Use 'mse', 'normalized_mse', or 'mae'."
        )


# ---------------------------------------------------------------------------
# MeTAL: Task-Adaptive Loss (Baik et al., ICCV 2021)
#
# Per-step loss networks + adapters for support and query.
# Reference: https://github.com/baiksung/MeTAL
# ---------------------------------------------------------------------------


def _z_normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-normalize a tensor (matches reference implementation)."""
    return (x - x.mean()) / (x.std() + 1e-12)


class StepLossNetwork(nn.Module):
    """
    Per-step loss network L_φ^j.

    2-layer MLP: input_dim → hidden → 1.
    Parameters get affine-transformed by StepLossAdapter; the transformed
    version is used via functional_forward().
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def functional_forward(
        self, x: torch.Tensor, params: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward with affine-transformed params [W1', b1', W2', b2']."""
        out = F.relu(F.linear(x, params[0], params[1]))
        out = F.linear(out, params[2], params[3])
        return out.squeeze(-1)


class StepAffineAdapter(nn.Module):
    """
    Per-step adapter g_ψ^j.

    Takes aggregate task state → affine transform coefficients for loss network.
    Identity at initialization (multiplier_bias, offset_bias = 0).
    """

    def __init__(self, tau_dim: int, n_loss_params: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(tau_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_loss_params * 2)
        self.multiplier_bias = nn.Parameter(torch.zeros(n_loss_params))
        self.offset_bias = nn.Parameter(torch.zeros(n_loss_params))

    def forward(
        self, tau: torch.Tensor, loss_params: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Produce affine-transformed loss network params."""
        h = F.relu(self.fc1(tau))
        out = self.fc2(h)
        gen_mult, gen_offset = torch.chunk(out, 2, dim=-1)

        return [
            (
                (1 + self.multiplier_bias[i] * gen_mult[i]) * p
                + self.offset_bias[i] * gen_offset[i]
            )
            for i, p in enumerate(loss_params)
        ]


class MeTALModule(nn.Module):
    """
    MeTAL: Task-Adaptive Loss (Baik et al. 2021).

    Per-step loss networks + affine adapters for support and query.
    Inner loop gradient = standard_support_loss + meta_support_loss + meta_query_loss.
    Outer loop loss stays standard (no MeTAL involved).

    Call support_step() and query_step() from the inner loop — they handle
    all internal wiring (task state, normalization, adapter, functional forward).
    """

    def __init__(
        self,
        n_steps: int,
        n_base_params: int,
        output_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_steps = n_steps

        # Dimensions
        support_adapter_dim = 1 + n_base_params
        support_loss_dim = support_adapter_dim + (2 * output_dim)
        query_per_sample_dim = n_base_params + output_dim + 1
        n_loss_params = 4  # W1, b1, W2, b2 per StepLossNetwork

        # Per-step networks (stored as typed lists, registered via add_module)
        self._support_loss: List[StepLossNetwork] = []
        self._support_affine: List[StepAffineAdapter] = []
        self._query_loss: List[StepLossNetwork] = []
        self._query_affine: List[StepAffineAdapter] = []

        for i in range(n_steps):
            sl = StepLossNetwork(support_loss_dim, hidden_dim)
            sa = StepAffineAdapter(support_adapter_dim, n_loss_params, hidden_dim)
            ql = StepLossNetwork(query_per_sample_dim, hidden_dim)
            qa = StepAffineAdapter(query_per_sample_dim, n_loss_params, hidden_dim)

            self.add_module(f"support_loss_{i}", sl)
            self.add_module(f"support_affine_{i}", sa)
            self.add_module(f"query_loss_{i}", ql)
            self.add_module(f"query_affine_{i}", qa)

            self._support_loss.append(sl)
            self._support_affine.append(sa)
            self._query_loss.append(ql)
            self._query_affine.append(qa)

    def support_step(
        self,
        step_idx: int,
        fmodel: nn.Module,
        support_pred: torch.Tensor,
        support_y: torch.Tensor,
        cost_function: Callable,
    ) -> torch.Tensor:
        """
        Compute meta support loss for one inner step.

        Aggregate τ → normalize → adapter → transform loss params
        → per-sample τ (unnormalized agg expanded + preds + targets) → normalize → loss net → mean
        """
        # Aggregate task state
        loss = cost_function(support_pred, support_y)  # you get the loss
        param_means = torch.stack([p.mean() for p in fmodel.parameters()])
        tau_agg = torch.cat(
            [loss.unsqueeze(0), param_means], dim=0
        )  # you build a loss with auxilliary task information

        # Adapter operates on normalized aggregate
        adapted_params = self._support_affine[step_idx](
            _z_normalize(tau_agg),
            list(self._support_loss[step_idx].parameters()),
        )

        # Loss network operates on normalized per-sample state
        tau_per_sample = torch.cat(
            [
                tau_agg.unsqueeze(0).expand(support_pred.size(0), -1),
                support_pred,
                support_y,
            ],
            dim=-1,
        )

        return (
            self._support_loss[step_idx]
            .functional_forward(_z_normalize(tau_per_sample), adapted_params)
            .mean()
        )

    def query_step(
        self,
        step_idx: int,
        fmodel: nn.Module,
        query_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute meta query loss for one inner step.

        Per-sample query state → normalize → mean for adapter
        → adapter → transform loss params → loss net on per-sample → mean
        """
        # Per-sample query state: [param_means expanded, predictions, pred_norm²]
        param_means = torch.stack([p.mean() for p in fmodel.parameters()])
        pred_norm_sq = (query_pred**2).sum(dim=-1, keepdim=True)
        query_state = torch.cat(
            [
                param_means.unsqueeze(0).expand(query_pred.size(0), -1),
                query_pred,
                pred_norm_sq,
            ],
            dim=-1,
        )
        query_state_norm = _z_normalize(query_state)

        # Adapter on aggregate (mean across samples)
        adapted_params = self._query_affine[step_idx](
            query_state_norm.mean(dim=0),
            list(self._query_loss[step_idx].parameters()),
        )

        return (
            self._query_loss[step_idx]
            .functional_forward(query_state_norm, adapted_params)
            .mean()
        )

    def log_loss(
        self,
        step_idx: int,
        fmodel: nn.Module,
        support_pred: torch.Tensor,
        support_y: torch.Tensor,
        query_pred: torch.Tensor,
        cost_function: Callable,
    ) -> float:
        """Compute combined meta loss for logging (no grad)."""
        step_idx = min(step_idx, self.n_steps - 1)
        ms = self.support_step(step_idx, fmodel, support_pred, support_y, cost_function)
        mq = self.query_step(step_idx, fmodel, query_pred)
        return (ms + mq).item()


@dataclass
class MAMLConfig:
    """
    Configuration for MAML training.

    Hyperparameter defaults follow Finn et al. 2017 and experiment_bible.md.
    """

    # Inner loop (task adaptation)
    inner_lr: float = 0.01  # α: learning rate for task adaptation
    inner_steps: int = 1  # Number of gradient steps per task

    # Outer loop (meta-update)
    outer_lr: float = 0.001  # β: learning rate for meta-update
    meta_batch_size: int = 4  # Tasks per meta-update

    # Support/query split
    k_shot: int = 100  # Support set size for inner loop
    query_size: int = 1000  # Query set size for meta-gradient

    # Training loop
    max_outer_iterations: int = 10000
    patience: int = 50  # Iterations before validation check
    log_interval: int = 10

    # FOMAML (first-order approximation - faster, no second derivatives)
    first_order: bool = False

    # Device and reproducibility
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 42

    # LR scheduler
    warmup_iterations: int = 0  # Linear warmup iterations (0 = no warmup)
    use_scheduler: bool = False  # Use cosine annealing after warmup
    min_lr: float = 1e-6  # Minimum LR at end of cosine decay
    scheduler_type: str = "cosine"  # 'cosine' (single decay) or 'warm_restarts'
    T_0: int = 500  # Initial restart period (warm_restarts only)
    T_mult: int = 2  # Period multiplier after each restart

    # Loss function: 'mse', 'normalized_mse', 'mae'
    loss_function: str = "normalized_mse"

    # MeTAL (Baik et al. ICCV 2021): task-adaptive inner-loop loss
    metal_enabled: bool = False
    metal_hidden_dim: int = 64

    # Spectral structural loss via NUFFT
    spectral_loss_enabled: bool = False
    spectral_loss_mode_size: int = 64


class MAMLTrainer:
    """
    MAML trainer for meta-learning PDE operator initialization.

    Implements the MAML algorithm with:
    - Inner loop: K-shot adaptation on support set
    - Outer loop: Meta-gradient update on query set losses
    - Two-level early stopping: train patience → validation check
    """

    def __init__(
        self,
        model: nn.Module,
        config: MAMLConfig,
        train_loader: MetaLearningDataLoader,
        val_loader: Optional[MetaLearningDataLoader] = None,
    ):
        """
        Initialize MAML trainer.

        Args:
            model: PDE operator network to meta-train
            config: MAML hyperparameters
            train_loader: Data loader for meta-training tasks
            val_loader: Optional data loader for meta-validation (early stopping)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Move model to device and store
        self.model = model.to(config.device)
        self.device = config.device

        # Pointwise loss — used as base term in cost function
        loss_type = config.loss_function
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

        # MeTAL: per-step task-adaptive loss (Baik et al. 2021)
        self.metal: Optional[MeTALModule] = None

        if config.metal_enabled:
            n_base_params = sum(1 for _ in model.parameters())
            output_dim = list(model.parameters())[-1].shape[0]

            self.metal = MeTALModule(
                n_steps=config.inner_steps,
                n_base_params=n_base_params,
                output_dim=output_dim,
                hidden_dim=config.metal_hidden_dim,
            ).to(config.device)

            n_metal_params = sum(p.numel() for p in self.metal.parameters())
            print(
                f"  MeTAL enabled: {config.inner_steps} steps × (support + query) networks"
            )
            print(f"  Base model params: {n_base_params}, output_dim: {output_dim}")
            print(f"  MeTAL total params: {n_metal_params:,}")

        # Bind inner step function — no conditionals in the hot loop
        self._inner_step = (
            self._metal_inner_step
            if self.metal is not None
            else self._standard_inner_step
        )

        # Outer loop optimizer (meta-update) — includes MeTAL params if enabled
        opt_params = list(self.model.parameters())
        if self.metal is not None:
            opt_params += list(self.metal.parameters())
        self.outer_opt = torch.optim.Adam(opt_params, lr=config.outer_lr)

        # LR scheduler: optional warmup → cosine decay (single or warm restarts)
        self.scheduler = None
        if config.use_scheduler or config.warmup_iterations > 0:
            schedulers = []
            milestones = []

            # Phase 1: linear warmup
            if config.warmup_iterations > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.outer_opt,
                    start_factor=1.0 / config.warmup_iterations,
                    end_factor=1.0,
                    total_iters=config.warmup_iterations,
                )
                schedulers.append(warmup)
                milestones.append(config.warmup_iterations)

            # Phase 2: cosine decay
            if config.use_scheduler:
                post_warmup_iters = (
                    config.max_outer_iterations - config.warmup_iterations
                )
                if config.scheduler_type == "warm_restarts":
                    decay = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.outer_opt,
                        T_0=config.T_0,
                        T_mult=config.T_mult,
                        eta_min=config.min_lr,
                    )
                else:
                    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.outer_opt,
                        T_max=post_warmup_iters,
                        eta_min=config.min_lr,
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

        # Training state
        self.iteration = 0
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_train_state = None  # Stash weights when train loss improves
        self.patience_counter = 0
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

    def cost_function(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        coords: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Unified cost: pointwise loss + spectral loss (when enabled + coords provided)."""
        pw = self._pointwise_loss(pred, target)
        if self.config.spectral_loss_enabled and coords is not None:
            x_pts, y_pts = coords
            spec = compute_spectral_loss(
                pred,
                target,
                x_pts,
                y_pts,
                self._current_Lx,
                self._current_Ly,
                self.config.spectral_loss_mode_size,
            )
            return pw + spec
        return pw

    # ------------------------------------------------------------------
    # Inner step implementations (bound in __init__, no conditionals)
    # ------------------------------------------------------------------

    def _standard_inner_step(
        self,
        fmodel: nn.Module,
        diffopt: object,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
        query_x: torch.Tensor,
        step_idx: int,
    ) -> None:
        """Standard MAML inner step: gradient on configured loss."""
        support_pred = fmodel(support_x)
        support_loss = self.cost_function(support_pred, support_y, support_coords)
        diffopt.step(support_loss)  # type: ignore[attr-defined]

    def _metal_inner_step(
        self,
        fmodel: nn.Module,
        diffopt: object,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        support_coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
        query_x: torch.Tensor,
        step_idx: int,
    ) -> None:
        """MeTAL inner step: support_loss + meta_support_loss + meta_query_loss."""
        assert self.metal is not None

        support_pred = fmodel(support_x)
        support_loss = self.cost_function(support_pred, support_y, support_coords)
        meta_support = self.metal.support_step(
            step_idx, fmodel, support_pred, support_y, self._pointwise_loss
        )

        query_pred = fmodel(query_x)
        meta_query = self.metal.query_step(step_idx, fmodel, query_pred)

        diffopt.step(support_loss + meta_support + meta_query)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Core MAML
    # ------------------------------------------------------------------

    def compute_task(self, task: PDETask, seed: int) -> torch.Tensor:
        """
        Compute query loss after inner loop adaptation.

        This is the core MAML operation:
        1. Get support/query split from task
        2. Adapt model on support set (inner loop)
        3. Evaluate adapted model on query set
        4. Return query loss (gradient flows back to meta-parameters)

        Args:
            task: PDE task with support/query data
            seed: Random seed for reproducible support/query split

        Returns:
            Query loss tensor (differentiable w.r.t. meta-parameters)
        """
        # Get support/query split
        support, query, support_coords, query_coords = task.get_support_query_split(
            K_shot=self.config.k_shot,
            query_size=self.config.query_size,
            seed=seed,
        )

        if self.config.inner_steps == 0:
            raise ValueError("You cannot run training with inner_steps == 0")

        # Unpack tensors (already on device from task loader)
        support_x, support_y = support
        query_x, query_y = query

        # Set domain info for spectral loss
        self._current_Lx = task.Lx
        self._current_Ly = task.Ly

        # Inner loop optimizer (recreated each task)
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.config.inner_lr)

        # Inner loop with higher library
        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,
            track_higher_grads=not self.config.first_order,
        ) as (fmodel, diffopt):
            # Pre-adaptation loss (logging only)
            with torch.no_grad():
                pre_pred = fmodel(support_x)
                pre_loss = self.cost_function(pre_pred, support_y, support_coords)
                pre_metal = ""
                if self.metal is not None:
                    pre_query = fmodel(query_x)
                    ml = self.metal.log_loss(
                        0, fmodel, pre_pred, support_y, pre_query, self._pointwise_loss
                    )
                    pre_metal = f", metal_loss={ml:.6f}"
                print(f"\t\tpre-adapt: support_loss={pre_loss.item():.6f}{pre_metal}")

            # Inner loop: adapt on support set
            for j in range(self.config.inner_steps):
                self._inner_step(
                    fmodel, diffopt, support_x, support_y, support_coords, query_x, j
                )

            # Post-adaptation loss (logging only)
            with torch.no_grad():
                post_pred = fmodel(support_x)
                post_loss = self.cost_function(post_pred, support_y, support_coords)
                post_metal = ""
                if self.metal is not None:
                    post_query = fmodel(query_x)
                    ml = self.metal.log_loss(
                        self.config.inner_steps - 1,
                        fmodel,
                        post_pred,
                        support_y,
                        post_query,
                        self._pointwise_loss,
                    )
                    post_metal = f", metal_loss={ml:.6f}"
                print(
                    f"\t\tpost-adapt: support_loss={post_loss.item():.6f}{post_metal}"
                    f" ({self.config.inner_steps} steps)"
                )

            # Evaluate adapted model on query set (standard loss — no MeTAL here)
            query_pred = fmodel(query_x)
            query_loss = self.cost_function(query_pred, query_y, query_coords)

        return query_loss

    def outer_step(self, tasks: List[PDETask]) -> float:
        """
        Perform one meta-update step.

        Args:
            tasks: Batch of tasks for this meta-update

        Returns:
            Average query loss across tasks
        """
        self.model.train()
        self.outer_opt.zero_grad()

        # Compute query losses for all tasks
        total_loss = torch.tensor(0.0, device=self.device)
        for i, task in enumerate(tasks):
            # Use iteration + task index as seed for reproducibility
            print(f"\ttask [{i}/{len(tasks)}]: {task.task_name}")
            seed = self.iteration * len(tasks) + i
            task_loss = self.compute_task(task, seed)
            print(f"\t\t\tloss={task_loss.item():.2f}")
            total_loss += task_loss

        # Average loss
        avg_loss = (
            total_loss / len(tasks)
        )  # Every task subgraph that was extended inner_step times converges right here. This is where they connect.

        # Meta-gradient update
        avg_loss.backward()
        self.outer_opt.step()

        return avg_loss.item()

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
            # Temporarily enable gradients for inner loop
            with torch.enable_grad():
                task_loss = self.compute_task(task, task_seed)
            total_loss += task_loss.item()

        return total_loss / len(tasks)

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
                f"  → Saved checkpoint to {checkpoint_dir / 'best_model.pt'}. Train loss: {winner_train_loss:.6f}, Val loss: {winner_val_loss:.6f}"
            )
            print(f"      - Current weights train_loss: {train_loss}")
            print(f"      - Best-train weights train_loss: {self.best_train_loss}")

            # Continue training from winner weights
            self.model.load_state_dict(winner_state)
            self.best_train_state = copy.deepcopy(winner_state)
            self.best_train_loss = winner_train_loss  # Reset train baseline

            self.save_checkpoint(checkpoint_dir / "best_model.pt")
            self.save_checkpoint(checkpoint_dir / "latest_model.pt")
            return True
        else:
            print("  → Validation stalled. Early stopping.")
            self.save_checkpoint(checkpoint_dir / "latest_model.pt")
            return False

    def train(
        self, checkpoint_dir: Optional[Path] = None, log_interval: Optional[int] = None
    ) -> Dict[str, List]:
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

        print("Starting MAML training:")
        print(f"  Device: {self.device}")
        print(f"  Inner LR (α): {self.config.inner_lr}")
        print(f"  Outer LR (β): {self.config.outer_lr}")
        print(f"  Inner steps: {self.config.inner_steps}")
        print(f"  Meta batch size: {self.config.meta_batch_size}")
        print(f"  K-shot: {self.config.k_shot}")
        print(f"  Query size: {self.config.query_size}")
        print(f"  FOMAML: {self.config.first_order}")
        if self.config.warmup_iterations > 0:
            print(f"  Warmup: {self.config.warmup_iterations} iterations")
        if self.config.use_scheduler:
            sched_desc = f"cosine annealing to {self.config.min_lr}"
            if self.config.scheduler_type == "warm_restarts":
                sched_desc += f" (warm restarts: T_0={self.config.T_0}, T_mult={self.config.T_mult})"
            print(f"  LR scheduler: {sched_desc}")
        if self._resumed:
            print(f"  Resuming from iteration {start_iteration}")
        print()

        early_stopped: bool = False
        train_loss: float = 0.0
        iteration = start_iteration
        self._stop_requested = False

        # Catch Ctrl+C as a stop request — save happens at the iteration boundary
        prev_handler = signal.signal(
            signal.SIGINT, lambda *_: setattr(self, "_stop_requested", True)
        )

        for iteration in range(start_iteration, self.config.max_outer_iterations):
            self.iteration = iteration

            # Sample task batch
            tasks = self.train_loader.sample_batch(
                self.config.meta_batch_size, seed=iteration
            )

            # Meta-update
            train_loss = self.outer_step(tasks)

            # Step LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["iteration"].append(iteration)

            # Level 1: Track train loss improvement and stash best weights
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.best_train_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Logging
            if iteration % log_interval == 0:
                lr_str = (
                    f", lr={self.outer_opt.param_groups[0]['lr']:.2e}"
                    if self.scheduler
                    else ""
                )
                if self.val_loader:
                    print(
                        f"Iter {iteration:5d}: train_loss={train_loss:.6f}, patience={self.patience_counter}{lr_str}"
                    )
                else:
                    print(f"Iter {iteration:5d}: train_loss={train_loss:.6f}{lr_str}")

            # Level 2: Validation check when patience exhausted
            if (
                self.patience_counter >= self.config.patience
                and self.val_loader is not None
            ):
                if self.validate(train_loss, checkpoint_dir):
                    self.patience_counter = 0
                else:
                    early_stopped = True
                    break

            # Graceful shutdown — all iteration state is consistent here
            if self._stop_requested:
                print(
                    f"\n  Stop requested after iteration {iteration}. Saving checkpoint..."
                )
                self.save_checkpoint(checkpoint_dir / "latest_model.pt")
                print(
                    f"  Saved to {checkpoint_dir / 'latest_model.pt'}. Resume with --resume."
                )
                signal.signal(signal.SIGINT, prev_handler)
                return self.history

        # Restore original handler
        signal.signal(signal.SIGINT, prev_handler)

        if self.val_loader is not None and not early_stopped:
            print("Doing end-of-loop validation run.")
            self.validate(train_loss, checkpoint_dir)
        else:
            # Ensure model has best weights regardless of how we exited
            if self.best_train_state is not None:
                self.model.load_state_dict(self.best_train_state)

            # Always save final model (in case validation never triggered)
            if not (checkpoint_dir / "best_model.pt").exists():
                print(
                    "Saving final model as best_model.pt (no validation checkpoint was saved)..."
                )
                self.save_checkpoint(checkpoint_dir / "best_model.pt")

        print()
        print(f"Training complete at iteration {iteration}")
        print(f"  Best train loss: {self.best_train_loss:.6f}")
        if self.val_loader is not None:
            print(f"  Best val loss: {self.best_val_loss:.6f}")

        return self.history

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
            },
            path,
        )

        # MeTAL: save entire module as single file alongside main checkpoint
        if self.metal is not None:
            torch.save(self.metal.state_dict(), path.parent / "metal_state.pt")

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
            torch.random.set_rng_state(rng_cpu)
        rng_cuda = checkpoint.get("rng_state_cuda")
        if rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_cuda)

        # Restore training loop state
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.history = checkpoint.get(
            "history", {"train_loss": [], "val_loss": [], "iteration": []}
        )
        self._resumed = True

        # MeTAL: load module state if it exists
        metal_state_path = path.parent / "metal_state.pt"
        if self.metal is not None and metal_state_path.exists():
            self.metal.load_state_dict(
                torch.load(
                    metal_state_path, map_location=self.device, weights_only=True
                )
            )
            print(f"  Loaded MeTAL module from {metal_state_path}")

        print(f"Loaded checkpoint from {path}")
        print(f"  Iteration: {self.iteration}")
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
