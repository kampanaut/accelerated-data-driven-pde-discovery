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
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Type
import numpy as np
import torch

from src.data.fourier_eval import build_wavenumbers, fourier_eval_2d


@dataclass
class CoefficientSpec:
    """Specification for one coefficient to extract via JVP.

    Attributes:
        name: Human-readable name for this coefficient (e.g. "D_u", "nu_v").
              Used as key in results dicts and npz files.
        perturb_indices: Which input features to perturb together.
                         E.g. [4, 5] for u_xx + u_yy in a 10-input PDE.
        output_index: Which output component to read the response from.
                      E.g. 0 for u_t, 1 for v_t.
        true_value: Ground-truth coefficient for error computation.
        coeff_name: The physical coefficient being recovered (e.g. "nu", "D_u").
                    Specs sharing a coeff_name are overlaid in histograms.
                    Defaults to name (each estimate is its own coefficient).
    """

    name: str
    perturb_indices: list[int]
    output_index: int
    true_value: float
    coeff_name: str = ""

    def __post_init__(self) -> None:
        if not self.coeff_name:
            self.coeff_name = self.name


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
    n_features: int = 10  # Override in subclass for scalar PDEs (e.g., heat: 5)
    n_targets: int = 2  # Override in subclass for scalar PDEs (e.g., heat: 1)

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
    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject proportional noise into features and recompute targets.

        Targets are recomputed from noisy features via the PDE RHS so that
        noise propagates self-consistently through the physics. For PDEs
        where targets can't be recomputed (e.g. NS with implicit pressure),
        independent proportional noise is added to targets instead.

        Args:
            features: (N, n_features) float32
            targets: (N, n_targets) float32
            noise_level: Proportional noise level
            generator: Optional seeded generator for reproducibility

        Returns:
            (noisy_features, recomputed_targets)
        """
        pass

    @property
    @abstractmethod
    def diffusion_coeffs(self) -> dict:
        """Return diffusion coefficients as dict."""
        pass

    @property
    @abstractmethod
    def coefficient_specs(self) -> list[CoefficientSpec]:
        """Return CoefficientSpec list for JVP-based coefficient extraction."""
        pass

    def get_support_query_split(
        self,
        K_shot: int,
        query_size: int,
        seed: Optional[int] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate support/query data at random collocation points.
        Returns clean data — noise injection is the caller's responsibility.
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
            (n_total, self.n_features), dtype=torch.float32, device=self.device
        )
        all_targets = torch.empty(
            (n_total, self.n_targets), dtype=torch.float32, device=self.device
        )

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

    Stores u_hat/v_hat FFT coefficients. Targets computed from PDE RHS
    (self-consistent with features).

    Diffusion coefficients: D_u, D_v
    Reaction coefficients: k1, k2
    """

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(
            data["u_hat"], dtype=torch.complex128, device='cpu'
        )
        self.v_hat = torch.tensor(
            data["v_hat"], dtype=torch.complex128, device='cpu'
        )

    def _extract_pde_params(self) -> None:
        self.D_u = self.simulation_params.get("D_u") or self.ic_config.get("D_u_used")
        self.D_v = self.simulation_params.get("D_v") or self.ic_config.get("D_v_used")
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
        _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y, self.device)

        u_hat = self.u_hat[snap_idx].to(device=self.device)
        v_hat = self.v_hat[snap_idx].to(device=self.device)

        ikx = 1j * self.kx.unsqueeze(0)         # (1, nx)
        iky = 1j * self.ky.unsqueeze(1)         # (ny, 1)
        neg_kx2 = -(self.kx.unsqueeze(0)) ** 2  # (1, nx)
        neg_ky2 = -(self.ky.unsqueeze(1)) ** 2  # (ny, 1)

        u = _eval(u_hat)
        v = _eval(v_hat)
        u_x = _eval(ikx * u_hat)
        u_y = _eval(iky * u_hat)
        v_x = _eval(ikx * v_hat)
        v_y = _eval(iky * v_hat)
        u_xx = _eval(neg_kx2 * u_hat)
        u_yy = _eval(neg_ky2 * u_hat)
        v_xx = _eval(neg_kx2 * v_hat)
        v_yy = _eval(neg_ky2 * v_hat)

        features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1).to(device=self.device)

        u_sq_v = u ** 2 * v
        u_t = self.D_u * (u_xx + u_yy) + self.k1 - (self.k2 + 1) * u + u_sq_v
        v_t = self.D_v * (v_xx + v_yy) + self.k2 * u - u_sq_v
        targets = torch.stack([u_t, v_t], dim=1)

        return features.float(), targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + noise

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

    @property
    def coefficient_specs(self) -> list[CoefficientSpec]:
        return [
            CoefficientSpec(
                name="D_u", perturb_indices=[4, 5], output_index=0, true_value=self.D_u
            ),
            CoefficientSpec(
                name="D_v", perturb_indices=[8, 9], output_index=1, true_value=self.D_v
            ),
        ]


class FitzHughNagumoTask(PDETask):
    """
    FitzHugh-Nagumo PDE task (Fourier-native).

    Stores u_hat/v_hat FFT coefficients. Targets computed from PDE RHS
    (self-consistent with features, like Brusselator).

    Diffusion coefficients: D_u (activator), D_v (recovery)
    Kinetic parameters: eps, a, b
    """

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(
            data["u_hat"], dtype=torch.complex128, device='cpu'
        )
        self.v_hat = torch.tensor(
            data["v_hat"], dtype=torch.complex128, device='cpu'
        )

    def _extract_pde_params(self) -> None:
        self.b = self.simulation_params.get("b", 0.0)

        D_u = self.simulation_params.get("D_u")
        D_v = self.simulation_params.get("D_v")
        eps = self.simulation_params.get("eps")
        a = self.simulation_params.get("a")

        if D_u is None:
            D_u = self.ic_config.get("D_u_used")
        if D_v is None:
            D_v = self.ic_config.get("D_v_used")
        if eps is None:
            eps = self.ic_config.get("eps")
        if a is None:
            a = self.ic_config.get("a_used")

        if D_u is None or D_v is None or eps is None or a is None:
            missing = [
                k
                for k, v in {"D_u": D_u, "D_v": D_v, "eps": eps, "a": a}.items()
                if v is None
            ]
            raise ValueError(f"FHN coefficients missing: {missing}")

        self.D_u = D_u
        self.D_v = D_v
        self.eps = eps
        self.a = a

        self.D_u = float(self.D_u)
        self.D_v = float(self.D_v)
        self.eps = float(self.eps)
        self.a = float(self.a)
        self.b = float(self.b)

    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y, self.device)

        u_hat = self.u_hat[snap_idx].to(device=self.device)
        v_hat = self.v_hat[snap_idx].to(device=self.device)

        ikx = 1j * self.kx.unsqueeze(0)         # (1, nx)
        iky = 1j * self.ky.unsqueeze(1)         # (ny, 1)
        neg_kx2 = -(self.kx.unsqueeze(0)) ** 2  # (1, nx)
        neg_ky2 = -(self.ky.unsqueeze(1)) ** 2  # (ny, 1)

        u = _eval(u_hat)
        v = _eval(v_hat)
        u_x = _eval(ikx * u_hat)
        u_y = _eval(iky * u_hat)
        v_x = _eval(ikx * v_hat)
        v_y = _eval(iky * v_hat)
        u_xx = _eval(neg_kx2 * u_hat)
        u_yy = _eval(neg_ky2 * u_hat)
        v_xx = _eval(neg_kx2 * v_hat)
        v_yy = _eval(neg_ky2 * v_hat)

        features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

        u_t = self.D_u * (u_xx + u_yy) + u - u ** 3 - v
        v_t = self.D_v * (v_xx + v_yy) + self.eps * (u - self.a * v - self.b)
        targets = torch.stack([u_t, v_t], dim=1)

        return features.float(), targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + noise

        # Recompute targets from noisy features via FHN RHS
        u, v = features[:, 0], features[:, 1]
        u_xx, u_yy = features[:, 4], features[:, 5]
        v_xx, v_yy = features[:, 8], features[:, 9]
        u_t = self.D_u * (u_xx + u_yy) + u - u**3 - v
        v_t = self.D_v * (v_xx + v_yy) + self.eps * (u - self.a * v - self.b)
        targets = torch.stack([u_t, v_t], dim=1)

        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"D_u": self.D_u, "D_v": self.D_v}

    @property
    def coefficient_specs(self) -> list[CoefficientSpec]:
        return [
            CoefficientSpec(
                name="D_u", perturb_indices=[4, 5], output_index=0, true_value=self.D_u
            ),
            CoefficientSpec(
                name="D_v", perturb_indices=[8, 9], output_index=1, true_value=self.D_v
            ),
        ]


class LambdaOmegaTask(PDETask):
    """
    Lambda-Omega PDE task (Fourier-native).

    Stores u_hat/v_hat FFT coefficients. Targets computed from PDE RHS
    (self-consistent with features, like FHN/Brusselator).

    Diffusion coefficients: D_u, D_v
    Kinetic parameters: a, c
    """

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(
            data["u_hat"], dtype=torch.complex128, device='cpu'
        )
        self.v_hat = torch.tensor(
            data["v_hat"], dtype=torch.complex128, device='cpu'
        )

    def _extract_pde_params(self) -> None:
        D_u = self.simulation_params.get("D_u")
        D_v = self.simulation_params.get("D_v")
        a = self.simulation_params.get("a")
        c = self.simulation_params.get("c")

        if D_u is None:
            D_u = self.ic_config.get("D_u_used")
        if D_v is None:
            D_v = self.ic_config.get("D_v_used")
        if a is None:
            a = self.ic_config.get("a")
        if c is None:
            c = self.ic_config.get("c_used")

        if D_u is None or D_v is None or a is None or c is None:
            missing = [
                k
                for k, v in {"D_u": D_u, "D_v": D_v, "a": a, "c": c}.items()
                if v is None
            ]
            raise ValueError(f"Lambda-Omega coefficients missing: {missing}")

        self.D_u = float(D_u)
        self.D_v = float(D_v)
        self.a = float(a)
        self.c = float(c)

    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y, self.device)

        u_hat = self.u_hat[snap_idx].to(device=self.device)
        v_hat = self.v_hat[snap_idx].to(device=self.device)

        ikx = 1j * self.kx.unsqueeze(0)         # (1, nx)
        iky = 1j * self.ky.unsqueeze(1)         # (ny, 1)
        neg_kx2 = -(self.kx.unsqueeze(0)) ** 2  # (1, nx)
        neg_ky2 = -(self.ky.unsqueeze(1)) ** 2  # (ny, 1)

        u = _eval(u_hat)
        v = _eval(v_hat)
        u_x = _eval(ikx * u_hat)
        u_y = _eval(iky * u_hat)
        v_x = _eval(ikx * v_hat)
        v_y = _eval(iky * v_hat)
        u_xx = _eval(neg_kx2 * u_hat)
        u_yy = _eval(neg_ky2 * u_hat)
        v_xx = _eval(neg_kx2 * v_hat)
        v_yy = _eval(neg_ky2 * v_hat)

        features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

        r2 = u ** 2 + v ** 2
        u_t = self.D_u * (u_xx + u_yy) + self.a * u - (u + self.c * v) * r2
        v_t = self.D_v * (v_xx + v_yy) + self.a * v + (self.c * u - v) * r2
        targets = torch.stack([u_t, v_t], dim=1)

        return features.float(), targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + noise

        # Recompute targets from noisy features via Lambda-Omega RHS
        u, v = features[:, 0], features[:, 1]
        u_xx, u_yy = features[:, 4], features[:, 5]
        v_xx, v_yy = features[:, 8], features[:, 9]
        r2 = u**2 + v**2
        u_t = self.D_u * (u_xx + u_yy) + self.a * u - (u + self.c * v) * r2
        v_t = self.D_v * (v_xx + v_yy) + self.a * v + (self.c * u - v) * r2
        targets = torch.stack([u_t, v_t], dim=1)

        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"D_u": self.D_u, "D_v": self.D_v}

    @property
    def coefficient_specs(self) -> list[CoefficientSpec]:
        return [
            CoefficientSpec(
                name="D_u", perturb_indices=[4, 5], output_index=0, true_value=self.D_u
            ),
            CoefficientSpec(
                name="D_v", perturb_indices=[8, 9], output_index=1, true_value=self.D_v
            ),
        ]


class NavierStokesTask(PDETask):
    """
    Navier-Stokes PDE task (Fourier-native).

    Stores u_hat/v_hat/p_hat FFT coefficients. Targets computed from
    the NS momentum equation using pressure from Dedalus:
        u_t = -(u*u_x + v*u_y) - p_x + nu*(u_xx + u_yy)
        v_t = -(u*v_x + v*v_y) - p_y + nu*(v_xx + v_yy)

    Diffusion coefficient: nu (kinematic viscosity)
    """

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(
            data["u_hat"], dtype=torch.complex128, device='cpu'
        )
        self.v_hat = torch.tensor(
            data["v_hat"], dtype=torch.complex128, device='cpu'
        )
        self.p_hat = torch.tensor(
            data["p_hat"], dtype=torch.complex128, device='cpu'
        )

    def _extract_pde_params(self) -> None:
        self.nu = float(self.simulation_params.get("nu") or self.ic_config.get("nu"))
        if self.nu is None:
            raise Exception("NS coefficient nu is None")

    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y, self.device)

        u_hat = self.u_hat[snap_idx].to(device=self.device)
        v_hat = self.v_hat[snap_idx].to(device=self.device)
        p_hat = self.p_hat[snap_idx].to(device=self.device)

        ikx = 1j * self.kx.unsqueeze(0)         # (1, nx)
        iky = 1j * self.ky.unsqueeze(1)         # (ny, 1)
        neg_kx2 = -(self.kx.unsqueeze(0)) ** 2  # (1, nx)
        neg_ky2 = -(self.ky.unsqueeze(1)) ** 2  # (ny, 1)

        u = _eval(u_hat)
        v = _eval(v_hat)
        u_x = _eval(ikx * u_hat)
        u_y = _eval(iky * u_hat)
        v_x = _eval(ikx * v_hat)
        v_y = _eval(iky * v_hat)
        u_xx = _eval(neg_kx2 * u_hat)
        u_yy = _eval(neg_ky2 * u_hat)
        v_xx = _eval(neg_kx2 * v_hat)
        v_yy = _eval(neg_ky2 * v_hat)

        # Pressure gradient from p_hat
        p_x = _eval(ikx * p_hat)
        p_y = _eval(iky * p_hat)

        features = torch.stack([u, v, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy], dim=1)

        # NS momentum equation
        u_t = -(u * u_x + v * u_y) - p_x + self.nu * (u_xx + u_yy)
        v_t = -(u * v_x + v * v_y) - p_y + self.nu * (v_xx + v_yy)
        targets = torch.stack([u_t, v_t], dim=1)

        return features.float(), targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Proportional noise on features
        feat_std = features.std(dim=0, keepdim=True)
        feat_noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + feat_noise

        # Can't recompute targets from noisy features (p_hat is in Fourier
        # space, not derivable from noisy pointwise features).
        # Add proportional noise to targets independently.
        tgt_std = targets.std(dim=0, keepdim=True)
        tgt_noise = torch.randn(
            targets.shape,
            dtype=targets.dtype,
            device=targets.device,
            generator=generator,
        ) * (noise_level * tgt_std)
        targets = targets + tgt_noise

        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"nu": self.nu}

    @property
    def coefficient_specs(self) -> list[CoefficientSpec]:
        return [
            CoefficientSpec(
                name="nu_u",
                perturb_indices=[4, 5],
                output_index=0,
                true_value=self.nu,
                coeff_name="nu",
            ),
            CoefficientSpec(
                name="nu_v",
                perturb_indices=[8, 9],
                output_index=1,
                true_value=self.nu,
                coeff_name="nu",
            ),
        ]


class HeatEquationTask(PDETask):
    """
    Heat equation task (Fourier-native, scalar field).

    u_t = D * nabla^2(u)

    Stores u_hat FFT coefficients only. 5 features, 1 target.
    Diffusion coefficient: D
    """

    n_features: int = 5
    n_targets: int = 1

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(
            data["u_hat"], dtype=torch.complex128, device='cpu'
        )

    def _extract_pde_params(self) -> None:
        D = self.simulation_params.get("D")
        if D is None:
            D = self.ic_config.get("D_used")
        if D is None:
            raise ValueError(
                "Heat equation coefficient D missing from simulation_params and ic_config"
            )
        self.D = float(D)

    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y, self.device)

        u_hat = self.u_hat[snap_idx].to(device=self.device)

        ikx = 1j * self.kx.unsqueeze(0)         # (1, nx)
        iky = 1j * self.ky.unsqueeze(1)         # (ny, 1)
        neg_kx2 = -(self.kx.unsqueeze(0)) ** 2  # (1, nx)
        neg_ky2 = -(self.ky.unsqueeze(1)) ** 2  # (ny, 1)

        u = _eval(u_hat)
        u_x = _eval(ikx * u_hat)
        u_y = _eval(iky * u_hat)
        u_xx = _eval(neg_kx2 * u_hat)
        u_yy = _eval(neg_ky2 * u_hat)

        features = torch.stack([u, u_x, u_y, u_xx, u_yy], dim=1)
        u_t = self.D * (u_xx + u_yy)
        targets = torch.stack([u_t], dim=1)

        return features.float(), targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + noise

        # Recompute targets from noisy features
        u_xx, u_yy = features[:, 3], features[:, 4]
        u_t = self.D * (u_xx + u_yy)
        targets = u_t.unsqueeze(1)

        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"D": self.D}

    @property
    def coefficient_specs(self) -> list[CoefficientSpec]:
        return [
            CoefficientSpec(
                name="D", perturb_indices=[3, 4], output_index=0, true_value=self.D
            ),
        ]


class NLHeatEquationTask(PDETask):
    """
    Nonlinear heat equation task (Fourier-native, scalar field).

    u_t = K * (1 - u) * nabla^2(u)

    Stores u_hat FFT coefficients only. 5 features, 1 target.
    Diffusion coefficient: K
    """

    n_features: int = 5
    n_targets: int = 1

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(
            data["u_hat"], dtype=torch.complex128, device='cpu'
        )

    def _extract_pde_params(self) -> None:
        K = self.simulation_params.get("K")
        if K is None:
            K = self.ic_config.get("K_used")
        if K is None:
            raise ValueError(
                "Nonlinear heat coefficient K missing from simulation_params and ic_config"
            )
        self.K = float(K)

    def _evaluate_snapshot(
        self,
        snap_idx: int,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _eval = lambda fh: fourier_eval_2d(fh, E_x, E_y, self.device)

        u_hat = self.u_hat[snap_idx].to(device=self.device)

        ikx = 1j * self.kx.unsqueeze(0)         # (1, nx)
        iky = 1j * self.ky.unsqueeze(1)         # (ny, 1)
        neg_kx2 = -(self.kx.unsqueeze(0)) ** 2  # (1, nx)
        neg_ky2 = -(self.ky.unsqueeze(1)) ** 2  # (ny, 1)

        u = _eval(u_hat)
        u_x = _eval(ikx * u_hat)
        u_y = _eval(iky * u_hat)
        u_xx = _eval(neg_kx2 * u_hat)
        u_yy = _eval(neg_ky2 * u_hat)

        features = torch.stack([u, u_x, u_y, u_xx, u_yy], dim=1)
        u_t = self.K * (1 - u) * (u_xx + u_yy)
        targets = torch.stack([u_t], dim=1)

        return features.float(), targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + noise

        # Recompute targets from noisy features
        u = features[:, 0]
        u_xx, u_yy = features[:, 3], features[:, 4]
        u_t = self.K * (1 - u) * (u_xx + u_yy)
        targets = u_t.unsqueeze(1)

        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"K": self.K}

    @property
    def coefficient_specs(self) -> list[CoefficientSpec]:
        return [
            CoefficientSpec(
                name="K", perturb_indices=[3, 4], output_index=0, true_value=self.K
            ),
        ]


TASK_REGISTRY: dict[str, type[PDETask]] = {
    "br": BrusselatorTask,
    "fhn": FitzHughNagumoTask,
    "lo": LambdaOmegaTask,
    "ns": NavierStokesTask,
    "heat": HeatEquationTask,
    "nl_heat": NLHeatEquationTask,
}


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

    def sample_batch(self, n_tasks: int, seed: Optional[int] = None) -> List[PDETask]:
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
