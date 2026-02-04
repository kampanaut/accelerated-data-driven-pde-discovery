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
from typing import List, Optional, Dict, Any
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from .task_loader import PDETask, NavierStokesTask, BrusselatorTask, MetaLearningDataLoader


@dataclass
class MAMLConfig:
    """
    Configuration for MAML training.

    Hyperparameter defaults follow Finn et al. 2017 and experiment_bible.md.
    """
    # Inner loop (task adaptation)
    inner_lr: float = 0.01           # α: learning rate for task adaptation
    inner_steps: int = 1             # Number of gradient steps per task

    # Outer loop (meta-update)
    outer_lr: float = 0.001          # β: learning rate for meta-update
    meta_batch_size: int = 4         # Tasks per meta-update

    # Support/query split
    k_shot: int = 100                # Support set size for inner loop
    query_size: int = 1000           # Query set size for meta-gradient

    # Noise (for robustness experiments)
    noise_level: float = 0.0         # 0.0 = clean data

    # Training loop
    max_outer_iterations: int = 10000
    patience: int = 50               # Iterations before validation check
    log_interval: int = 10

    # FOMAML (first-order approximation - faster, no second derivatives)
    first_order: bool = False

    # Device and reproducibility
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 42


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
        val_loader: Optional[MetaLearningDataLoader] = None
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

        # Outer loop optimizer (meta-update)
        self.outer_opt = torch.optim.Adam(
            self.model.parameters(),
            lr=config.outer_lr
        )

        # Training state
        self.iteration = 0
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_train_state = None  # Stash weights when train loss improves

    def compute_task_loss(self, task: PDETask, seed: int) -> torch.Tensor:
        """
        Compute query loss after inner loop adaptation.

        This is the core MAML operation:
        1. Get support/query split from task
        2. Adapt model on support set (inner loop)
        3. Evaluate adapted model on query set
        4. Return query loss (gradient flows back to meta-parameters)

        Args:
            task: Navier-Stokes task with support/query data
            seed: Random seed for reproducible support/query split

        Returns:
            Query loss tensor (differentiable w.r.t. meta-parameters)
        """
        # Get support/query split
        support, query = task.get_support_query_split(
            K_shot=self.config.k_shot,
            query_size=self.config.query_size,
            seed=seed,
            noise_level=self.config.noise_level
        )

        # Convert to tensors
        support_x = torch.tensor(support[0], dtype=torch.float32, device=self.device)
        support_y = torch.tensor(support[1], dtype=torch.float32, device=self.device)
        query_x = torch.tensor(query[0], dtype=torch.float32, device=self.device)
        query_y = torch.tensor(query[1], dtype=torch.float32, device=self.device)

        # Inner loop optimizer (recreated each task)
        inner_opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.inner_lr
        )

        # Inner loop with higher library
        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            copy_initial_weights=False,  # Share meta-parameters
            track_higher_grads=not self.config.first_order  # FOMAML skips second derivatives
        ) as (fmodel, diffopt):

            # Inner loop: adapt on support set
            for _ in range(self.config.inner_steps):
                support_pred = fmodel(support_x)
                support_loss = F.mse_loss(support_pred, support_y)
                diffopt.step(support_loss)

            # Evaluate adapted model on query set
            query_pred = fmodel(query_x)
            query_loss = F.mse_loss(query_pred, query_y)

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
        total_loss = 0.0
        for i, task in enumerate(tasks):
            # Use iteration + task index as seed for reproducibility
            seed = self.iteration * len(tasks) + i
            task_loss = self.compute_task_loss(task, seed)
            total_loss += task_loss

        # Average loss
        avg_loss = total_loss / len(tasks)

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

        # Actually we need gradients for inner loop even during eval
        for i, task in enumerate(tasks):
            task_seed = seed + i
            # Temporarily enable gradients for inner loop
            with torch.enable_grad():
                task_loss = self.compute_task_loss(task, task_seed)
            total_loss += task_loss.item()

        return total_loss / len(tasks)

    def validate(self, train_loss: float, checkpoint_dir: Path, history: Dict[str, List]) -> bool:
        print(f"  → Patience exhausted. Comparing current vs best-train on validation...")

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
            print(f"  → Winner: current weights")
        else:
            winner_state = self.best_train_state
            winner_val_loss = best_train_val_loss
            winner_train_loss = self.best_train_loss
            print(f"  → Winner: best-train weights")

        history['val_loss'].append(winner_val_loss)

        # Did winner beat previous best val?
        if winner_val_loss < self.best_val_loss:
            self.best_val_loss = winner_val_loss
            print(f"  → Validation improved! Resetting patience.")

            print(f"  → Saved checkpoint to {checkpoint_dir / 'best_model.pt'}. Train loss: {winner_train_loss:.6f}, Val loss: {winner_val_loss:.6f}")
            print(f"      - Current weights train_loss: {train_loss}")
            print(f"      - Best-train weights train_loss: {self.best_train_loss}")

            # Continue training from winner weights
            self.model.load_state_dict(winner_state)
            self.best_train_state = copy.deepcopy(winner_state)
            self.best_train_loss = winner_train_loss # Reset train baseline

            self.save_checkpoint(checkpoint_dir / 'best_model.pt')
            return True
        else:
            print(f"  → Validation stalled. Early stopping.")
            return False

    def train(
        self,
        checkpoint_dir: Optional[Path] = None,
        log_interval: Optional[int] = None
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

        # Initialize tracking
        patience_counter = 0
        history: Dict[str, List] = {
            'train_loss': [],
            'val_loss': [],
            'iteration': []
        }

        print(f"Starting MAML training:")
        print(f"  Device: {self.device}")
        print(f"  Inner LR (α): {self.config.inner_lr}")
        print(f"  Outer LR (β): {self.config.outer_lr}")
        print(f"  Inner steps: {self.config.inner_steps}")
        print(f"  Meta batch size: {self.config.meta_batch_size}")
        print(f"  K-shot: {self.config.k_shot}")
        print(f"  Query size: {self.config.query_size}")
        print(f"  Noise level: {self.config.noise_level}")
        print(f"  FOMAML: {self.config.first_order}")
        print()

        early_stopped: bool = False
        train_loss: float = 0.0
        for iteration in range(self.config.max_outer_iterations):
            self.iteration = iteration

            # Sample task batch
            tasks = self.train_loader.sample_batch(
                self.config.meta_batch_size,
                seed=iteration
            )

            # Meta-update
            train_loss = self.outer_step(tasks)

            # Record history
            history['train_loss'].append(train_loss)
            history['iteration'].append(iteration)

            # Level 1: Track train loss improvement and stash best weights
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.best_train_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if iteration % log_interval == 0:
                if self.val_loader: 
                    print(f"Iter {iteration:5d}: train_loss={train_loss:.6f}, patience={patience_counter}")
                else:
                    print(f"Iter {iteration:5d}: train_loss={train_loss:.6f}")

            # Level 2: Validation check when patience exhausted
            if patience_counter >= self.config.patience and self.val_loader is not None:
                if self.validate(train_loss, checkpoint_dir, history):
                    patience_counter = 0
                else:
                    early_stopped = True
                    break

            # Early stop if no val_loader and patience exhausted
            # if patience_counter >= self.config.patience and self.val_loader is None:
            #     print(f"Patience exhausted (no val_loader). Stopping at iteration {iteration}.")
            #     # Load best-train weights so fallback save gets the right weights
            #     if self.best_train_state is not None:
            #         self.model.load_state_dict(self.best_train_state)
            #         print(f"  → Loaded best-train weights for saving.")
            #     break
            # No, we can't early stop. When there's no validation set, then we
            # just train until the end.

        if self.val_loader is not None and not early_stopped:
            print("Doing end-of-loop validation run.")
            self.validate(train_loss, checkpoint_dir, history)
        else:
            # Ensure model has best weights regardless of how we exited
            if self.best_train_state is not None:
                self.model.load_state_dict(self.best_train_state)


            # Always save final model (in case validation never triggered)
            if not (checkpoint_dir / 'best_model.pt').exists():
                print("Saving final model as best_model.pt (no validation checkpoint was saved)...")
                self.save_checkpoint(checkpoint_dir / 'best_model.pt')

        print()
        print(f"Training complete at iteration {iteration}")
        print(f"  Best train loss: {self.best_train_loss:.6f}")
        if self.val_loader is not None:
            print(f"  Best val loss: {self.best_val_loss:.6f}")

        return history

    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.outer_opt.state_dict(),
            'config': self.config,
            'iteration': self.iteration,
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss,
            'best_train_state': self.best_train_state,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.outer_opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint.get('iteration', 0)
        self.best_train_loss = checkpoint.get('best_train_loss', float('inf'))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_train_state = checkpoint.get('best_train_state', None)

        print(f"Loaded checkpoint from {path}")
        print(f"  Iteration: {self.iteration}")
        print(f"  Best train loss: {self.best_train_loss:.6f}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")


def get_meta_learned_init(
    checkpoint_path: Path,
    model_class: type = None,
    **model_kwargs
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
    # Import here to avoid circular dependency
    if model_class is None:
        from ..networks.pde_operator_network import PDEOperatorNetwork
        model_class = PDEOperatorNetwork

    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def fine_tune(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    lr: float,
    max_steps: int,
    device: str = 'cpu'
) -> List[float]:
    """
    Fine-tune model on given data and return loss curve.

    Used for evaluation: compare convergence of MAML init vs random init.

    Args:
        model: Model to fine-tune (will be modified in-place)
        features: Input features array (N, input_dim)
        targets: Target outputs array (N, output_dim)
        lr: Learning rate for fine-tuning
        max_steps: Number of gradient steps
        device: Device to run on

    Returns:
        List of loss values at each step
    """
    model = model.to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device)

    losses = []
    for _ in range(max_steps):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    return losses


def compare_initializations(
    maml_checkpoint: Path,
    task: PDETask,
    k_shot: int,
    max_steps: int,
    lr: float,
    seed: int = 42,
    device: str = 'cpu',
    model_class: type = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Compare MAML initialization vs random initialization on a task.

    Args:
        maml_checkpoint: Path to MAML checkpoint
        task: Task to evaluate on
        k_shot: Number of samples for fine-tuning
        max_steps: Number of fine-tuning steps
        lr: Learning rate for fine-tuning
        seed: Random seed for data sampling
        device: Device to run on
        model_class: Model class (default: PDEOperatorNetwork)
        **model_kwargs: Model constructor arguments

    Returns:
        Dict with 'maml_losses', 'baseline_losses', 'samples'
    """
    if model_class is None:
        from ..networks.pde_operator_network import PDEOperatorNetwork
        model_class = PDEOperatorNetwork

    # Sample data ONCE (fair comparison)
    support, _ = task.get_support_query_split(
        K_shot=k_shot,
        query_size=0,  # Only need support for fine-tuning
        seed=seed
    )
    features, targets = support

    # MAML initialization
    maml_model = get_meta_learned_init(maml_checkpoint, model_class, **model_kwargs)
    maml_model = copy.deepcopy(maml_model)  # Don't modify checkpoint model
    maml_losses = fine_tune(maml_model, features, targets, lr, max_steps, device)

    # Random initialization (baseline)
    baseline_model = model_class(**model_kwargs)
    baseline_losses = fine_tune(baseline_model, features, targets, lr, max_steps, device)

    return {
        'maml_losses': maml_losses,
        'baseline_losses': baseline_losses,
        'k_shot': k_shot,
        'max_steps': max_steps,
        'lr': lr,
        'task_name': task.task_name
    }
