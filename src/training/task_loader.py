"""
Task data loaders for MAML-based meta-learning on Navier-Stokes PDEs.

Loads pre-computed derivative data from .npz files and provides support/query
splits for meta-learning.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np


class NavierStokesTask:
    """
    Single meta-learning task representing one Navier-Stokes initial condition.

    Each task contains:
    - Features: (u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy)
    - Targets: (u_t, v_t)
    - Metadata: IC configuration, viscosity, etc.

    The task provides support/query splits for MAML:
    - Support set (D): K samples for inner loop adaptation
    - Query set (D'): Q samples for meta-gradient evaluation
    """

    def __init__(self, npz_path: Path):
        """
        Load task from .npz file.

        Args:
            npz_path: Path to .npz file with pre-computed derivatives

        Raises:
            ValueError: If data is corrupted (NaN/Inf/shape mismatches)
        """
        self.npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)

        # Stack features (10 input features)
        self.features = np.stack([
            data['u'].flatten(),
            data['v'].flatten(),
            data['u_x'].flatten(),
            data['u_y'].flatten(),
            data['u_xx'].flatten(),
            data['u_yy'].flatten(),
            data['v_x'].flatten(),
            data['v_y'].flatten(),
            data['v_xx'].flatten(),
            data['v_yy'].flatten(),
        ], axis=1)  # Shape: (N_samples, 10)

        # Stack targets (2 output features)
        self.targets = np.stack([
            data['u_t'].flatten(),
            data['v_t'].flatten(),
        ], axis=1)  # Shape: (N_samples, 2)

        # Extract metadata
        self.ic_config = data['ic_config'].item() if 'ic_config' in data else {}
        self.nu = float(data['nu_used']) if 'nu_used' in data else None
        self.n_samples = len(self.features)
        self.task_name = self.npz_path.stem

        # Strict validation - fail loudly on corrupted data
        self._validate()

    def _validate(self):
        """
        Validate data integrity.

        Raises:
            ValueError: If metadata indicates failure, or data has shape mismatches
        """
        # Check metadata file for status (authoritative source)
        txt_path = self.npz_path.with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                content = f.read()
            if 'Status: FAILED' in content:
                raise ValueError(f"Task marked as failed in metadata: {self.npz_path.name}")

        # Check shape consistency
        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError(
                f"Shape mismatch in {self.npz_path.name}: "
                f"features {self.features.shape} vs targets {self.targets.shape}"
            )

    def get_support_query_split(
        self,
        K_shot: int,
        query_size: int,
        seed: Optional[int] = None,
        noise_level: float = 0.0
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Split task data into support set (D) and query set (D') for MAML.

        MAML terminology:
        - Support set (D): K samples used for inner loop adaptation (task training)
        - Query set (D'): Q samples used for meta-gradient evaluation (task testing)
        - D and D' must be disjoint (no overlap between support and query)

        Args:
            K_shot: Support set size (number of samples for inner loop)
            query_size: Query set size (number of samples for meta-gradient)
            seed: Random seed for reproducible splits
            noise_level: Noise level to use (0.0 = clean data, >0 loads from *_noisy.npz)

        Returns:
            support: Tuple of (features[K], targets[K]) for inner loop
            query: Tuple of (features[Q], targets[Q]) for meta-gradient

        Raises:
            ValueError: If K_shot + query_size exceeds task size
            FileNotFoundError: If noise_level > 0 but noisy file doesn't exist
        """
        # Get features and targets (clean or noisy)
        if noise_level > 0.0:
            features, targets = self._load_noisy_data(noise_level)
        else:
            features, targets = self.features, self.targets

        n_samples = len(features)

        if K_shot + query_size > n_samples:
            raise ValueError(
                f"K_shot ({K_shot}) + query_size ({query_size}) = {K_shot + query_size} "
                f"exceeds task size ({n_samples}) for {self.task_name}"
            )

        # Shuffle indices
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)

        # Split into support and query (disjoint sets)
        support_idx = indices[:K_shot]
        query_idx = indices[K_shot:K_shot + query_size]

        support = (features[support_idx], targets[support_idx])
        query = (features[query_idx], targets[query_idx])

        return support, query

    def _load_noisy_data(
        self,
        noise_level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load noisy features and targets from *_noisy.npz file.

        Args:
            noise_level: Noise level to load (e.g., 0.01, 0.05, 0.10)

        Returns:
            Tuple of (features, targets) arrays with noise applied

        Raises:
            FileNotFoundError: If noisy file doesn't exist
            KeyError: If noise level not found in noisy file
        """
        noisy_path = self.npz_path.parent / f"{self.npz_path.stem}_noisy.npz"

        if not noisy_path.exists():
            raise FileNotFoundError(
                f"Noisy file not found: {noisy_path}. "
                f"Run scripts/inject_noise.py to create it."
            )

        noisy_data = np.load(noisy_path, allow_pickle=True)
        level_key = f"noise_{noise_level:.2f}"

        # Stack features (10 input features)
        try:
            features = np.stack([
                noisy_data[f'{level_key}/u'].flatten(),
                noisy_data[f'{level_key}/v'].flatten(),
                noisy_data[f'{level_key}/u_x'].flatten(),
                noisy_data[f'{level_key}/u_y'].flatten(),
                noisy_data[f'{level_key}/u_xx'].flatten(),
                noisy_data[f'{level_key}/u_yy'].flatten(),
                noisy_data[f'{level_key}/v_x'].flatten(),
                noisy_data[f'{level_key}/v_y'].flatten(),
                noisy_data[f'{level_key}/v_xx'].flatten(),
                noisy_data[f'{level_key}/v_yy'].flatten(),
            ], axis=1)

            # Stack targets (2 output features)
            targets = np.stack([
                noisy_data[f'{level_key}/u_t'].flatten(),
                noisy_data[f'{level_key}/v_t'].flatten(),
            ], axis=1)
        except KeyError as e:
            available = [k.split('/')[0] for k in noisy_data.keys() if '/' in k]
            available = sorted(set(available))
            raise KeyError(
                f"Noise level '{level_key}' not found in {noisy_path}. "
                f"Available levels: {available}"
            ) from e

        return features, targets

    def __repr__(self) -> str:
        """String representation of task."""
        return (
            f"NavierStokesTask(\n"
            f"  name='{self.task_name}',\n"
            f"  samples={self.n_samples:,},\n"
            f"  nu={self.nu:.6f if self.nu else 'unknown'},\n"
            f"  ic_type='{self.ic_config.get('type', 'unknown')}'\n"
            f")"
        )


class MetaLearningDataLoader:
    """
    Manages multiple Navier-Stokes tasks for MAML meta-learning.

    Loads all .npz files from a directory, each representing a different task
    (typically different initial conditions with the same PDE parameters).

    Provides functionality for:
    - Loading all tasks from a directory
    - Sampling random task batches for meta-training
    - Splitting tasks into train/test sets
    """

    def __init__(self, data_dir: Path, task_pattern: str = "*.npz"):
        """
        Initialize data loader by loading all tasks from directory.

        Args:
            data_dir: Directory containing .npz task files
            task_pattern: Glob pattern for task files (default: "*.npz")

        Raises:
            ValueError: If no .npz files found or if any task is corrupted
        """
        self.data_dir = Path(data_dir)
        npz_files = sorted(self.data_dir.glob(task_pattern))

        # Exclude noisy files (they're loaded on-demand by get_support_query_split)
        npz_files = [f for f in npz_files if not f.stem.endswith('_noisy')]

        if len(npz_files) == 0:
            raise ValueError(f"No .npz files found in {data_dir} with pattern '{task_pattern}'")

        # Load all tasks, skipping invalid ones
        self.tasks: List[NavierStokesTask] = []
        self.task_names: List[str] = []
        skipped: List[str] = []

        print(f"Loading tasks from {data_dir}...")
        for f in npz_files:
            try:
                task = NavierStokesTask(f)
                self.tasks.append(task)
                self.task_names.append(f.stem)
            except ValueError as e:
                skipped.append(f"{f.stem}: {e}")

        if skipped:
            print(f"\n⚠ Skipped {len(skipped)} invalid tasks:")
            for msg in skipped:
                print(f"  {msg}")

        print(f"\n✓ Loaded {len(self.tasks)} tasks:")
        for name, task in zip(self.task_names, self.tasks):
            nu_str = f"ν={task.nu:.6f}" if task.nu else "ν=unknown"
            print(f"  {name:30s}  {task.n_samples:>7,} samples  {nu_str}")

    def sample_batch(self, n_tasks: int, seed: Optional[int] = None) -> List[NavierStokesTask]:
        """
        Sample random subset of tasks for meta-training batch.

        This is used in the MAML outer loop to get a batch of tasks for
        computing the meta-gradient.

        Args:
            n_tasks: Number of tasks to sample (meta-batch size)
            seed: Random seed for reproducibility

        Returns:
            List of n_tasks NavierStokesTask objects

        Raises:
            ValueError: If n_tasks exceeds number of available tasks
        """
        if n_tasks > len(self.tasks):
            raise ValueError(
                f"Cannot sample {n_tasks} tasks from {len(self.tasks)} available tasks"
            )

        rng = np.random.RandomState(seed)
        indices = rng.choice(len(self.tasks), size=n_tasks, replace=False)
        return [self.tasks[i] for i in indices]

    def get_task_by_name(self, name: str) -> NavierStokesTask:
        """
        Get specific task by name for evaluation.

        Args:
            name: Task name (stem of .npz filename)

        Returns:
            NavierStokesTask object

        Raises:
            ValueError: If task name not found
        """
        try:
            idx = self.task_names.index(name)
            return self.tasks[idx]
        except ValueError:
            raise ValueError(
                f"Task '{name}' not found. Available tasks: {self.task_names}"
            )

    def train_test_split(
        self,
        test_ratio: float = 0.2,
        seed: Optional[int] = None
    ) -> Tuple[List[NavierStokesTask], List[NavierStokesTask]]:
        """
        Split tasks into train/test sets for meta-learning.

        Meta-train tasks: Used for meta-training (learn meta-initialization θ)
        Meta-test tasks: Held-out for final evaluation (test generalization)

        Args:
            test_ratio: Fraction of tasks to reserve for testing (default: 0.2)
            seed: Random seed for reproducible splits

        Returns:
            train_tasks: List of tasks for meta-training
            test_tasks: List of held-out tasks for meta-testing
        """
        n_test = max(1, int(len(self.tasks) * test_ratio))

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.tasks))

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_tasks = [self.tasks[i] for i in train_idx]
        test_tasks = [self.tasks[i] for i in test_idx]

        print(f"\nTrain/test split:")
        print(f"  Train tasks: {len(train_tasks)}")
        print(f"  Test tasks: {len(test_tasks)}")

        return train_tasks, test_tasks

    def __len__(self) -> int:
        """Number of tasks."""
        return len(self.tasks)

    def __repr__(self) -> str:
        """String representation of data loader."""
        return (
            f"MetaLearningDataLoader(\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  num_tasks={len(self.tasks)},\n"
            f"  task_names={self.task_names[:3] + ['...'] if len(self.tasks) > 3 else self.task_names}\n"
            f")"
        )
