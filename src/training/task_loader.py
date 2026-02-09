"""
Task data loaders for MAML-based meta-learning on PDEs.

Fourier-native architecture: tasks store FFT coefficients and evaluate
features/targets at random collocation points on-the-fly on GPU. This gives:
- Spectrally exact derivatives (no finite-difference error)
- Fresh random points each batch (infinite effective dataset)
- Arbitrary evaluation locations (not grid-locked)
- GPU-accelerated Fourier synthesis
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Optional, Type
import numpy as np
import torch

from src.data.fourier_eval import build_wavenumbers


class PDETask(ABC):
    """
    Abstract base class for Fourier-native PDE discovery tasks.

    Stores FFT coefficients and evaluates features at random collocation
    points on-the-fly. Subclasses implement PDE-specific coefficient loading,
    evaluation, and noise injection.

    The task provides support/query splits for MAML:
    - Support set (D): K samples for inner loop adaptation
    - Query set (D'): Q samples for meta-gradient evaluation
    """

    ny: int
    nx: int
    n_snapshots: int


    def __init__(self, npz_path: Path, device: str = "cuda"):
        self.npz_path = Path(npz_path)
        self.device = device
        data = np.load(npz_path, allow_pickle=True)

        # Common metadata
        if "ic_config" in data:
            self.ic_config = data["ic_config"].item()
        else:
            raise Exception("`ic_config` key not included in 'data'")

        if "simulation_params" in data:
            self.simulation_params = data["simulation_params"].item()
        else:
            raise Exception("`simulation_params` key not included in 'data'")

        self.task_name = self.npz_path.stem

        # Domain geometry
        Lx, Ly = self.simulation_params["domain_size"]
        self.Lx, self.Ly = float(Lx), float(Ly)

        # Subclass loads PDE-specific Fourier tensors, sets n_snapshots/ny/nx
        self._load_coefficients(data)

        # Precompute wavenumbers on GPU
        self.kx, self.ky = build_wavenumbers(
            self.nx, self.ny, self.Lx, self.Ly, device=device
        )

        # n_samples for compatibility (snapshot_count * grid_size)
        self.n_samples = self.n_snapshots * self.ny * self.nx

        # Subclass-specific parameter extraction
        self._extract_pde_params()

    @abstractmethod
    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        """
        Load PDE-specific Fourier tensors to GPU.
        Must set self.n_snapshots, self.ny, self.nx.
        """
        pass

    @abstractmethod
    def _extract_pde_params(self) -> None:
        """Extract PDE-specific coefficients from metadata."""
        pass

    @abstractmethod
    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate features and targets for one snapshot at given points.

        Args:
            snap_idx: Snapshot index
            E_x: Phase matrix, shape (n_points, nx), complex128
            E_y: Phase matrix, shape (n_points, ny), complex128

        Returns:
            (features, targets): float32 tensors, shapes (n_points, 10) and (n_points, 2)
        """
        pass

    @abstractmethod
    def _inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        clean_targets: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject noise into features (and optionally targets).

        Args:
            features: (N, 10) float32
            targets: (N, 2) float32
            noise_level: Proportional noise level
            clean_targets: If True, don't modify targets

        Returns:
            (noisy_features, noisy_or_clean_targets)
        """
        pass

    @property
    @abstractmethod
    def diffusion_coeffs(self) -> dict:
        """Return diffusion coefficients as dict."""
        pass

    def get_support_query_split(
        self,
        K_shot: int,
        query_size: int,
        seed: Optional[int] = None,
        noise_level: float = 0.0,
        clean_targets: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate support/query data at random collocation points.
        Everything stays on GPU — no numpy, no host transfers.
        """
        n_total = K_shot + query_size

        # Torch RNG on device
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)

        # Sample random snapshot indices and spatial coordinates (all on GPU)
        snap_idx = torch.randint(
            0, self.n_snapshots, (n_total,), generator=gen, device=self.device
        )
        x_pts = (
            torch.rand(n_total, generator=gen, device=self.device, dtype=torch.float64)
            * self.Lx
        )
        y_pts = (
            torch.rand(n_total, generator=gen, device=self.device, dtype=torch.float64)
            * self.Ly
        )

        # Build phase matrices on GPU
        E_x = torch.exp(1j * torch.outer(x_pts, self.kx))  # (n_total, nx)
        E_y = torch.exp(1j * torch.outer(y_pts, self.ky))  # (n_total, ny)

        # Group points by snapshot for efficient evaluation
        all_features = torch.empty(
            (n_total, 10), dtype=torch.float32, device=self.device
        )
        all_targets = torch.empty((n_total, 2), dtype=torch.float32, device=self.device)

        unique_snaps = torch.unique(snap_idx)
        for si in unique_snaps:
            mask_idx = torch.where(snap_idx == si)[0]
            feats, tgts = self._evaluate_snapshot(
                si.item(),
                E_x[mask_idx],
                E_y[mask_idx],
            )
            all_features[mask_idx] = feats
            all_targets[mask_idx] = tgts

        # Inject noise if requested
        if noise_level > 0.0:
            all_features, all_targets = self._inject_noise(
                all_features, all_targets, noise_level, clean_targets
            )

        support = (all_features[:K_shot], all_targets[:K_shot])
        query = (all_features[K_shot:], all_targets[K_shot:])

        return support, query

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{self.task_name}',\n"
            f"  snapshots={self.n_snapshots}, grid=({self.ny}, {self.nx}),\n"
            f"  diffusion_coeffs={self.diffusion_coeffs},\n"
            f"  ic_type='{self.ic_config.get('type', 'unknown')}',\n"
            f"  device='{self.device}'\n"
            f")"
        )


class BrusselatorTask(PDETask):
    """
    Brusselator PDE task (Fourier-native).

    Stores A_hat/B_hat FFT coefficients. Targets computed from PDE RHS
    (self-consistent with features).

    Diffusion coefficients: D_u (for species A/u), D_v (for species B/v)
    Reaction coefficients: k1, k2
    """

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["A_hat"].shape[0]
        self.ny, self.nx = data["A_hat"].shape[1], data["A_hat"].shape[2]
        self.A_hat = torch.tensor(data["A_hat"], dtype=torch.complex128, device=self.device)
        self.B_hat = torch.tensor(data["B_hat"], dtype=torch.complex128, device=self.device)

    def _extract_pde_params(self) -> None:
        self.D_u = self.simulation_params.get("D_A") or self.ic_config.get("D_A_used")
        self.D_v = self.simulation_params.get("D_B") or self.ic_config.get("D_B_used")
        self.k1 = self.simulation_params.get("k1")
        self.k2 = self.simulation_params.get("k2")

        if self.D_u is None:
            raise Exception("BR coefficient D_u is None")
        if self.D_v is None:
            raise Exception("BR coefficient D_v is None")

        self.D_u = float(self.D_u)
        self.D_v = float(self.D_v)
        self.k1 = float(self.k1)
        self.k2 = float(self.k2)

    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from src.data.fourier_eval import evaluate_all_features

        return evaluate_all_features(
            self.A_hat[snap_idx],
            self.B_hat[snap_idx],
            self.kx,
            self.ky,
            E_x,
            E_y,
            self.D_u,
            self.D_v,
            self.k1,
            self.k2,
        )

    def _inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        clean_targets: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Proportional noise on features
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn_like(features) * (noise_level * feat_std)
        features = features + noise

        if not clean_targets:
            # Recompute targets from noisy features via PDE RHS (self-consistent)
            u, v = features[:, 0], features[:, 1]
            u_xx, u_yy = features[:, 4], features[:, 5]
            v_xx, v_yy = features[:, 8], features[:, 9]
            a_sq_b = u**2 * v
            u_t = self.D_u * (u_xx + u_yy) + self.k1 - (self.k2 + 1) * u + a_sq_b
            v_t = self.D_v * (v_xx + v_yy) + self.k2 * u - a_sq_b
            targets = torch.stack([u_t, v_t], dim=1)

        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"D_u": self.D_u, "D_v": self.D_v}


class NavierStokesTask(PDETask):
    """
    Navier-Stokes PDE task (Fourier-native).

    Stores u_hat/v_hat/u_t_hat/v_t_hat FFT coefficients. Targets come from
    temporal central differences (NOT from PDE RHS — pressure term is implicit).

    Diffusion coefficient: nu (kinematic viscosity)
    """

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(data["u_hat"], dtype=torch.complex128, device=self.device)
        self.v_hat = torch.tensor(data["v_hat"], dtype=torch.complex128, device=self.device)
        self.u_t_hat = torch.tensor(data["u_t_hat"], dtype=torch.complex128, device=self.device)
        self.v_t_hat = torch.tensor(data["v_t_hat"], dtype=torch.complex128, device=self.device)

    def _extract_pde_params(self) -> None:
        self.nu = float(
            self.simulation_params.get("nu") or self.ic_config.get("nu")
        )
        if self.nu is None:
            raise Exception("NS coefficient nu is None")

    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from src.data.fourier_eval import evaluate_ns_features

        return evaluate_ns_features(
            self.u_hat[snap_idx],
            self.v_hat[snap_idx],
            self.u_t_hat[snap_idx],
            self.v_t_hat[snap_idx],
            self.kx,
            self.ky,
            E_x,
            E_y,
        )

    def _inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        clean_targets: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Proportional noise on features
        feat_std = features.std(dim=0, keepdim=True)
        feat_noise = torch.randn_like(features) * (noise_level * feat_std)
        features = features + feat_noise

        if not clean_targets:
            # Can't recompute targets from noisy features (pressure is implicit).
            # Add proportional noise to targets independently.
            tgt_std = targets.std(dim=0, keepdim=True)
            tgt_noise = torch.randn_like(targets) * (noise_level * tgt_std)
            targets = targets + tgt_noise

        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"nu": self.nu}


class MetaLearningDataLoader:
    """
    Manages multiple PDE tasks for MAML meta-learning.

    Loads all Fourier .npz files from a directory, each representing a different
    task (typically different initial conditions with the same PDE parameters).
    """

    def __init__(
        self,
        data_dir: Path,
        task_class: Type[PDETask] = NavierStokesTask,
        task_pattern: str = "*_fourier.npz",
        device: str = "cuda",
    ):
        self.data_dir = Path(data_dir)
        self.task_class = task_class
        self.device = device
        npz_files = sorted(self.data_dir.glob(task_pattern))

        if len(npz_files) == 0:
            raise ValueError(
                f"No .npz files found in {data_dir} with pattern '{task_pattern}'"
            )

        # Load all tasks, skipping invalid ones
        self.tasks: List[PDETask] = []
        self.task_names: List[str] = []
        skipped: List[str] = []

        print(f"Loading tasks from {data_dir}...")
        for npz_path in npz_files:
            try:
                task = self.task_class(npz_path, device=self.device)
                self.tasks.append(task)
                self.task_names.append(npz_path.stem)
            except (ValueError, Exception) as e:
                skipped.append(f"{npz_path.stem}: {e}")

        if skipped:
            print(f"\n⚠ Skipped {len(skipped)} invalid tasks:")
            for msg in skipped:
                print(f"  {msg}")

        print(f"\n✓ Loaded {len(self.tasks)} tasks:")
        for name, task in zip(self.task_names, self.tasks):
            coeffs = task.diffusion_coeffs
            coeff_str = ", ".join(
                f"{k}={v:.6f}" if v else f"{k}=unknown" for k, v in coeffs.items()
            )
            print(f"  {name:30s}  {task.n_samples:>7,} samples  {coeff_str}")

    def sample_batch(
        self, n_tasks: int, seed: Optional[int] = None
    ) -> List[PDETask]:
        """
        Sample random subset of tasks for meta-training batch.
        """
        if n_tasks > len(self.tasks):
            raise ValueError(
                f"Cannot sample {n_tasks} tasks from {len(self.tasks)} available tasks"
            )

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        indices = torch.randperm(len(self.tasks), generator=gen)[:n_tasks]
        return [self.tasks[int(i)] for i in indices]

    def get_task_by_name(self, name: str) -> PDETask:
        """Get specific task by name for evaluation."""
        try:
            idx = self.task_names.index(name)
            return self.tasks[idx]
        except ValueError:
            raise ValueError(
                f"Task '{name}' not found. Available tasks: {self.task_names}"
            )

    def train_test_split(
        self, test_ratio: float = 0.2, seed: Optional[int] = None
    ) -> Tuple[List[PDETask], List[PDETask]]:
        """Split tasks into train/test sets for meta-learning."""
        n_test = max(1, int(len(self.tasks) * test_ratio))

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        indices = torch.randperm(len(self.tasks), generator=gen)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_tasks = [self.tasks[int(i)] for i in train_idx]
        test_tasks = [self.tasks[int(i)] for i in test_idx]

        print("\nTrain/test split:")
        print(f"  Train tasks: {len(train_tasks)}")
        print(f"  Test tasks: {len(test_tasks)}")

        return train_tasks, test_tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __repr__(self) -> str:
        return (
            f"MetaLearningDataLoader(\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  task_class={self.task_class.__name__},\n"
            f"  num_tasks={len(self.tasks)},\n"
            f"  task_names={self.task_names[:3] + ['...'] if len(self.tasks) > 3 else self.task_names}\n"
            f")"
        )
