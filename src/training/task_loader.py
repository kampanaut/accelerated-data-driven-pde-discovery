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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple, List, Optional, Type
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
    post_extract: Optional[Callable] = field(default=None, repr=False)  # (jvp_per_point, features) → corrected_per_point

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
    jacobian_plot_type: str = "histogram"  # Override in subclass (e.g., nl_heat: "scatter")

    def __init__(self, npz_path: Path, device: str = "cuda"):
        self.npz_path = Path(npz_path)
        self.device = device
        self.storage_device = "cpu"  # where *_hat tensors live; promoted to GPU if room
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

    def _hat_tensor_names(self) -> list[str]:
        """Return attribute names of *_hat Fourier coefficient tensors.

        Scans instance __dict__ for any attribute ending in '_hat'
        (e.g., u_hat, v_hat, p_hat). Used by hat_memory_bytes() and
        promote_storage() to operate generically across PDE types.
        """
        names = []
        for attr_name in self.__dict__:
            if attr_name.endswith("_hat"):
                names.append(attr_name)
        return names

    def hat_memory_bytes(self) -> int:
        """Total memory consumed by *_hat tensors in bytes."""
        total = 0
        for name in self._hat_tensor_names():
            tensor = getattr(self, name)
            total += tensor.nelement() * tensor.element_size()
        return total

    def promote_storage(self, device: str) -> None:
        """Move all *_hat tensors to the given device in-place.

        Called by MetaLearningDataLoader after loading all tasks,
        if enough VRAM is available. Avoids repeated CPU→GPU transfers
        during get_support_query_split().
        """
        for name in self._hat_tensor_names():
            tensor = getattr(self, name)
            setattr(self, name, tensor.to(device))
        self.storage_device = device

    @abstractmethod
    def evaluate_collocations(
        self,
        snap_idx_list: torch.Tensor,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Produce structural features and targets at collocation points.

        Args:
            snap_idx_list: Unique snapshot indices, shape (n_unique,)
            E_x: Masked x phase matrix, shape (n_unique, n_points, nx)
            E_y: Masked y phase matrix, shape (n_unique, n_points, ny)

        Returns:
            A tuple (features_list, targets):
              - features_list: list of float32 tensors, one per mixer
                (length equals n_outputs). Each tensor has shape
                (n_unique, n_points, n_features_for_that_mixer).
              - targets: float32 tensor of shape
                (n_unique, n_points, n_outputs) stacking per-mixer targets.

        The list-based feature output means each PDE owns its structural
        feature layout per mixer. There is no separate raw → structural
        transform step; subclasses compute structural features directly
        via their Fourier plumbing.
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

    @property
    @abstractmethod
    def rhs_feature_mask(self) -> list[bool]:
        """Boolean mask over input features: True = feature appears in PDE RHS.

        Length must equal n_features. Retained for documentation — the
        runtime masking path was deleted along with raw/zeroed mode.
        """
        pass

    # ── Mixer method API ─────────────────────────────────────────────
    # Abstract methods for the structural feature library + amortised
    # coefficient recovery framework. See docs/mixer_method.md.

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        """Number of output fields the mixer produces.

        Returns 1 for scalar PDEs (Heat, NLHeat, NS-vorticity) and 2 for
        paired PDEs (BR, FHN, λ-ω). Determines whether the composite
        operator network uses the one-head or two-head iMAML branch.
        """
        pass

    @property
    @abstractmethod
    def structural_feature_names(self) -> list[list[str]]:
        """Human-readable names for the structural features, per mixer.

        Outer list length equals n_outputs. Each inner list names the
        feature columns that mixer i consumes as input. For NLHeat:
            [['1-u', 'u_xx+u_yy']]
        For BR:
            [['u', 'u²v', 'u_xx+u_yy'],
             ['u', 'u²v', 'v_xx+v_yy']]
        """
        pass

    @property
    @abstractmethod
    def aux_loss_names(self) -> list[list[str]]:
        """Per-mixer aux loss coefficient names (matches auxiliary_losses keys).

        Outer list length equals n_outputs. Each inner list names the
        coefficients for which this mixer's `auxiliary_losses` returns a
        loss term. Empty inner list means that mixer has no aux losses.

        Used by `MixerNetwork.from_task` to pre-allocate Kendall
        log-variance parameters with the right keys.

        Examples:
            NLHeat: [['K']]
            Heat:   [['D_from_uxx', 'D_from_uyy']]
            BR:     [['D_u', 'k2', 'k1'], ['D_v', 'k2']]
        """
        pass

    @abstractmethod
    def extract_coefficients(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Extract recovered coefficient values from the adapted mixer.

        Runs Jacobian probes, regression slopes, and/or residual
        subtractions on the adapted composite network to recover the
        task-varying coefficients associated with this mixer.

        Args:
            mixer_idx: which output the mixer corresponds to (0 for u,
                       1 for v in paired PDEs).
            fast_model: the adapted composite operator network, exposes
                        forward_one(i, feats) for per-mixer forward.
            features: structural feature tensor for this mixer, shape
                      (N, len(structural_feature_names[mixer_idx])).

        Returns:
            {coeff_name: {'mean': scalar_tensor, 'std': scalar_tensor}, ...}
            One entry per task-varying coefficient recovered by this
            mixer. The 'std' is per-point dispersion (how linear the
            mixer is on the relevant feature, or regression residual
            std, depending on recovery type).
        """
        pass

    @abstractmethod
    def auxiliary_losses(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return auxiliary loss terms for the given mixer's equation.

        Used during training and evaluation inner loops (training-eval
        consistency). Each value is a normalized-MSE loss between a
        mixer-derived quantity and the task's ground-truth coefficient.
        Thin wrapper over extract_coefficients — uses the same JVP /
        regression / residual machinery, then wraps the extracted
        quantities against known truths from task metadata.

        Args:
            mixer_idx: which mixer (0 or 1).
            fast_model: the adapted composite operator network.
            features: structural feature tensor for this mixer.
            targets: ground-truth target tensor for this mixer's output.

        Returns:
            {coeff_name: tensor_loss, ...}. Empty dict if the PDE has no
            aux losses for this mixer (e.g., NLHeat clean run).
        """
        pass

    @abstractmethod
    def inject_noise_at_source(
        self,
        hat_tensors: dict[str, torch.Tensor],
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Inject source-level Gaussian noise into Fourier coefficients.

        Applied inside evaluate_collocations BEFORE derivative
        computation, so the derivatives inherit correlated noise from
        the source state. Matches SINDy's noise protocol and deployment
        reality.

        Args:
            hat_tensors: dict of hat tensors sliced to the current
                         batch, e.g. {'u_hat': shape (n_unique, ny, nx),
                         'v_hat': ...}.
            noise_level: proportional noise scale in [0, 1]. 0.0 is a
                         no-op and returns the inputs unchanged.
            generator: optional seeded torch.Generator for
                       reproducibility.

        Returns:
            Dict with the same keys and noisy versions of the tensors.
        """
        pass

    def get_support_query_split(
        self,
        K_shot: int,
        query_size: int,
        k_seed: Optional[int] = None,
        snapshot_seed: Optional[int] = None,
    ) -> Tuple[
        Tuple[list[torch.Tensor], torch.Tensor],
        Tuple[list[torch.Tensor], torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Generate support/query data at random collocation points.
        Returns clean data — noise injection is the caller's responsibility.
        Everything stays on GPU — no numpy, no host transfers.

        Returns:
            (support_feats_list, support_tgts),
            (query_feats_list, query_tgts),
            (support_x_pts, support_y_pts),
            (query_x_pts, query_y_pts)

        support_feats_list and query_feats_list are per-mixer lists, with
        length n_outputs. Each list element is a (N, n_features_for_that_mixer)
        tensor. Callers should index by mixer_idx to get a specific
        mixer's support/query features.
        """
        n_total = K_shot + query_size

        # Torch RNG on device
        k_shot_gen = torch.Generator(device=self.device)
        if k_seed is not None:
            k_shot_gen.manual_seed(k_seed)

        snapshot_gen = torch.Generator(device=self.storage_device)
        if snapshot_seed is not None:
            snapshot_gen.manual_seed(snapshot_seed)
        elif k_seed is not None:
            snapshot_gen.manual_seed(k_seed)
        elif self.storage_device == self.device:
            snapshot_gen = k_shot_gen


        # Sample random snapshot indices (uniform random — discrete dimension)
        snap_idx = torch.randint(
            0, self.n_snapshots, (n_total,), generator=snapshot_gen, device=self.storage_device
        )

        # Sobol quasi-random spatial coordinates for uniform coverage
        # (Wu et al. 2023: Sobol beats uniform random and LHS for PDE collocation)
        sobol_seed = (
            k_seed
            if k_seed is not None
            else int(torch.randint(0, 2**31, (1,), generator=k_shot_gen).item())
        )
        sobol = torch.quasirandom.SobolEngine(
            dimension=2, scramble=True, seed=sobol_seed
        )
        sobol_pts = sobol.draw(n_total)  # (n_total, 2), CPU float32
        x_pts = (sobol_pts[:, 0] * self.Lx).to(
            device=self.device, dtype=torch.float64
        )
        y_pts = (sobol_pts[:, 1] * self.Ly).to(
            device=self.device, dtype=torch.float64
        )

        # Build phase matrices on GPU
        E_x = torch.exp(1j * torch.outer(x_pts, self.kx))  # (n_total, nx)
        E_y = torch.exp(1j * torch.outer(y_pts, self.ky))  # (n_total, ny)

        unique_snaps, inverse = torch.unique(snap_idx, return_inverse=True)
        # sort_order and counts must be on self.device for indexing phase matrices
        inverse = inverse.to(self.device)
        sort_order = torch.argsort(inverse, stable=True)
        counts = torch.bincount(inverse)

        chunks_x = torch.split(E_x[sort_order], counts.tolist())
        chunks_y = torch.split(E_y[sort_order], counts.tolist())

        E_x_compact = torch.nn.utils.rnn.pad_sequence(
            list(chunks_x), batch_first=True
        )  # (len(unique_snaps), max(counts), nx)
        E_y_compact = torch.nn.utils.rnn.pad_sequence(
            list(chunks_y), batch_first=True
        )  # (len(unique_snaps), max(counts), ny)

        feats_list, tgts = self.evaluate_collocations(
            unique_snaps, E_x_compact, E_y_compact
        )

        mask_idx = torch.arange(0, E_x_compact.shape[1], device=self.device).unsqueeze(
            0
        ) < counts.unsqueeze(1)
        feats_list = [f[mask_idx] for f in feats_list]
        tgts = tgts[mask_idx]

        unsort = torch.argsort(sort_order)
        feats_list = [f[unsort] for f in feats_list]
        tgts = tgts[unsort]

        # Unsort coordinates the same way as features/targets
        x_pts = x_pts[unsort].float()
        y_pts = y_pts[unsort].float()

        support_feats = [f[:K_shot] for f in feats_list]
        query_feats = [f[K_shot:] for f in feats_list]
        support = (support_feats, tgts[:K_shot])
        query = (query_feats, tgts[K_shot:])
        support_coords = (x_pts[:K_shot], y_pts[:K_shot])
        query_coords = (x_pts[K_shot:], y_pts[K_shot:])

        return support, query, support_coords, query_coords

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
        self.u_hat = torch.tensor(data["u_hat"], dtype=torch.complex128, device=self.storage_device)
        self.v_hat = torch.tensor(data["v_hat"], dtype=torch.complex128, device=self.storage_device)

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

    def evaluate_collocations(
        self,
        snap_idx_list: torch.Tensor,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Produce structural features and targets for Brusselator.

        Mixer_u library: [u, u²v, u_xx+u_yy]
        Mixer_v library: [u, u²v, v_xx+v_yy]
        Targets: analytical u_t, v_t from the BR RHS.
        """
        neg_kx2 = -((self.kx.unsqueeze(0)) ** 2)  # (1, nx)
        neg_ky2 = -((self.ky.unsqueeze(1)) ** 2)  # (ny, 1)

        u_hat = self.u_hat[snap_idx_list].to(device=self.device)
        v_hat = self.v_hat[snap_idx_list].to(device=self.device)

        coeff_batch = torch.stack(
            [
                u_hat,
                v_hat,
                neg_kx2 * u_hat,  # u_xx
                neg_ky2 * u_hat,  # u_yy
                neg_kx2 * v_hat,  # v_xx
                neg_ky2 * v_hat,  # v_yy
            ],
            dim=0,
        )  # (6, n_unique, ny, nx)

        u, v, u_xx, u_yy, v_xx, v_yy = fourier_eval_2d(
            coeff_batch, E_x, E_y, self.device
        )

        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy
        u_sq_v = u * u * v

        mixer_u_features = torch.stack(
            [u, u_sq_v, lap_u], dim=2
        )  # (n_unique, n_pts, 3)
        mixer_v_features = torch.stack(
            [u, u_sq_v, lap_v], dim=2
        )  # (n_unique, n_pts, 3)

        u_t = self.D_u * lap_u + self.k1 - (self.k2 + 1) * u + u_sq_v
        v_t = self.D_v * lap_v + self.k2 * u - u_sq_v
        targets = torch.stack([u_t, v_t], dim=2)  # (n_unique, n_pts, 2)

        return [mixer_u_features.float(), mixer_v_features.float()], targets.float()

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
    def rhs_feature_mask(self) -> list[bool]:
        # BR RHS: D_u*∇²u + k1 - (k2+1)*u + u²v, D_v*∇²v + k2*u - u²v
        # Uses: u, v, u_xx, u_yy, v_xx, v_yy. NOT: u_x, u_y, v_x, v_y
        return [True, True, False, False, True, True, False, False, True, True]

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

    # ── Mixer method implementations ─────────────────────────────────

    @property
    def n_outputs(self) -> int:
        return 2

    @property
    def structural_feature_names(self) -> list[list[str]]:
        return [
            ["u", "u²v", "u_xx+u_yy"],
            ["u", "u²v", "v_xx+v_yy"],
        ]

    @property
    def aux_loss_names(self) -> list[list[str]]:
        return [
            ["D_u", "k2", "k1"],
            ["D_v", "k2"],
        ]

    def extract_coefficients(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Recover BR coefficients via JVP, algebra, and residual subtraction.

        mixer_u (idx=0) → {D_u, k2, k1}. mixer_v (idx=1) → {D_v, k2}.
        Both mixers produce a k2 estimate (cross-check handled downstream).
        """
        if mixer_idx not in (0, 1):
            raise ValueError(
                f"Brusselator has n_outputs=2; mixer_idx must be 0 or 1, got {mixer_idx}"
            )

        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(mixer_idx, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=False,
            retain_graph=False,
        )[0].detach()  # (N, 3)

        if mixer_idx == 0:
            jvp_u = grads[:, 0]            # ≈ -(k2 + 1)
            jvp_u_sq_v = grads[:, 1]       # ≈ +1 (fixed)
            jvp_lap_u = grads[:, 2]        # ≈ D_u
            D_u_mean = jvp_lap_u.mean()
            D_u_std = jvp_lap_u.std()
            k2_vals = -1.0 - jvp_u
            k2_mean = k2_vals.mean()
            k2_std = k2_vals.std()
            # Residual = mixer_output − Σ partial_i · feature_i ≈ k1.
            feats_det = features.detach()
            mixer_output = fast_model.forward_one(0, feats_det).detach()
            lin_combo = (
                jvp_u * feats_det[:, 0]
                + jvp_u_sq_v * feats_det[:, 1]
                + jvp_lap_u * feats_det[:, 2]
            )
            residual = mixer_output - lin_combo
            k1_mean = residual.mean()
            k1_std = residual.std()
            return {
                "D_u": {"mean": D_u_mean, "std": D_u_std},
                "k2": {"mean": k2_mean, "std": k2_std},
                "k1": {"mean": k1_mean, "std": k1_std},
            }

        # mixer_v (idx == 1)
        jvp_u = grads[:, 0]          # ≈ k2
        jvp_lap_v = grads[:, 2]      # ≈ D_v
        D_v_mean = jvp_lap_v.mean()
        D_v_std = jvp_lap_v.std()
        k2_mean = jvp_u.mean()
        k2_std = jvp_u.std()
        return {
            "D_v": {"mean": D_v_mean, "std": D_v_std},
            "k2": {"mean": k2_mean, "std": k2_std},
        }

    def auxiliary_losses(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Differentiable nMSE aux losses against ground-truth BR coefficients.

        Mirrors extract_coefficients but keeps the autograd graph alive.
        `targets` is unused — aux targets come from task metadata.
        """
        del targets  # unused; aux targets come from self.{D_u,D_v,k1,k2}
        if mixer_idx not in (0, 1):
            raise ValueError(
                f"Brusselator has n_outputs=2; mixer_idx must be 0 or 1, got {mixer_idx}"
            )

        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(mixer_idx, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 3)

        def _nmse(quantity: torch.Tensor, target_value: float) -> torch.Tensor:
            target_per_point = torch.full_like(quantity, target_value)
            diff = quantity - target_per_point
            mse = (diff * diff).mean()
            denom = (target_per_point * target_per_point).mean().clamp(min=1e-12)
            return mse / denom

        if mixer_idx == 0:
            jvp_u = grads[:, 0]
            jvp_u_sq_v = grads[:, 1]
            jvp_lap_u = grads[:, 2]
            aux_D_u = _nmse(jvp_lap_u, self.D_u)
            aux_k2 = _nmse(-1.0 - jvp_u, self.k2)
            # Residual for k1 — differentiable version uses features_grad.
            mixer_output = fast_model.forward_one(0, features_grad)
            lin_combo = (
                jvp_u * features_grad[:, 0]
                + jvp_u_sq_v * features_grad[:, 1]
                + jvp_lap_u * features_grad[:, 2]
            )
            residual = mixer_output - lin_combo
            aux_k1 = _nmse(residual, self.k1)
            return {"D_u": aux_D_u, "k2": aux_k2, "k1": aux_k1}

        # mixer_v (idx == 1)
        jvp_u = grads[:, 0]
        jvp_lap_v = grads[:, 2]
        aux_D_v = _nmse(jvp_lap_v, self.D_v)
        aux_k2 = _nmse(jvp_u, self.k2)
        return {"D_v": aux_D_v, "k2": aux_k2}

    def inject_noise_at_source(
        self,
        hat_tensors: dict[str, torch.Tensor],
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Real-space Gaussian noise via Fourier round-trip on u_hat and v_hat."""
        if noise_level <= 0.0:
            return hat_tensors
        u_hat = hat_tensors["u_hat"]
        v_hat = hat_tensors["v_hat"]

        u_real = torch.fft.ifft2(u_hat).real
        u_std = u_real.std()
        u_noise_real = torch.randn(
            u_real.shape,
            dtype=u_real.dtype,
            device=u_hat.device,
            generator=generator,
        ) * (noise_level * u_std)
        u_noise_hat = torch.fft.fft2(u_noise_real.to(dtype=u_hat.dtype))

        v_real = torch.fft.ifft2(v_hat).real
        v_std = v_real.std()
        v_noise_real = torch.randn(
            v_real.shape,
            dtype=v_real.dtype,
            device=v_hat.device,
            generator=generator,
        ) * (noise_level * v_std)
        v_noise_hat = torch.fft.fft2(v_noise_real.to(dtype=v_hat.dtype))

        return {"u_hat": u_hat + u_noise_hat, "v_hat": v_hat + v_noise_hat}


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
        self.u_hat = torch.tensor(data["u_hat"], dtype=torch.complex128, device=self.storage_device)
        self.v_hat = torch.tensor(data["v_hat"], dtype=torch.complex128, device=self.storage_device)

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

    def evaluate_collocations(
        self,
        snap_idx_list: torch.Tensor,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Produce structural features and targets for FitzHugh-Nagumo.

        Mixer_u library: [u_xx+u_yy, u, u³, v]
        Mixer_v library: [v_xx+v_yy, u, v]
        Targets: analytical u_t, v_t from the FHN RHS.
        """
        neg_kx2 = -((self.kx.unsqueeze(0)) ** 2)  # (1, nx)
        neg_ky2 = -((self.ky.unsqueeze(1)) ** 2)  # (ny, 1)

        u_hat = self.u_hat[snap_idx_list].to(device=self.device)
        v_hat = self.v_hat[snap_idx_list].to(device=self.device)

        coeff_batch = torch.stack(
            [
                u_hat,
                v_hat,
                neg_kx2 * u_hat,  # u_xx
                neg_ky2 * u_hat,  # u_yy
                neg_kx2 * v_hat,  # v_xx
                neg_ky2 * v_hat,  # v_yy
            ],
            dim=0,
        )  # (6, n_unique, ny, nx)

        u, v, u_xx, u_yy, v_xx, v_yy = fourier_eval_2d(
            coeff_batch, E_x, E_y, self.device
        )

        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy
        u_cubed = u * u * u

        mixer_u_features = torch.stack(
            [lap_u, u, u_cubed, v], dim=2
        )  # (n_unique, n_pts, 4)
        mixer_v_features = torch.stack(
            [lap_v, u, v], dim=2
        )  # (n_unique, n_pts, 3)

        u_t = self.D_u * lap_u + u - u_cubed - v
        v_t = self.D_v * lap_v + self.eps * (u - self.a * v - self.b)
        targets = torch.stack([u_t, v_t], dim=2)  # (n_unique, n_pts, 2)

        return [mixer_u_features.float(), mixer_v_features.float()], targets.float()

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
    def rhs_feature_mask(self) -> list[bool]:
        # FHN RHS: D_u*∇²u + u - u³ - v, D_v*∇²v + eps*(u - a*v - b)
        # Uses: u, v, u_xx, u_yy, v_xx, v_yy. NOT: u_x, u_y, v_x, v_y
        return [True, True, False, False, True, True, False, False, True, True]

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

    # ── Mixer method implementations ─────────────────────────────────

    @property
    def n_outputs(self) -> int:
        return 2

    @property
    def structural_feature_names(self) -> list[list[str]]:
        return [
            ["u_xx+u_yy", "u", "u³", "v"],
            ["v_xx+v_yy", "u", "v"],
        ]

    @property
    def aux_loss_names(self) -> list[list[str]]:
        return [
            ["D_u"],
            ["D_v", "eps", "eps_a", "eps_b"],
        ]

    def extract_coefficients(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Recover FHN coefficients via JVP and residual subtraction.

        mixer_u (idx=0) → {D_u}. mixer_v (idx=1) → {D_v, eps, eps_a, eps_b}
        using Framing A (compound coefficients); a and b are derived
        post-hoc at evaluation time.
        """
        if mixer_idx not in (0, 1):
            raise ValueError(
                f"FitzHugh-Nagumo has n_outputs=2; mixer_idx must be 0 or 1, got {mixer_idx}"
            )

        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(mixer_idx, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=False,
            retain_graph=False,
        )[0].detach()  # (N, n_cols)

        if mixer_idx == 0:
            jvp_lap_u = grads[:, 0]  # ≈ D_u
            D_u_mean = jvp_lap_u.mean()
            D_u_std = jvp_lap_u.std()
            return {"D_u": {"mean": D_u_mean, "std": D_u_std}}

        # mixer_v (idx == 1); library = [v_xx+v_yy, u, v]
        jvp_lap_v = grads[:, 0]  # ≈ D_v
        jvp_u = grads[:, 1]      # ≈ +eps
        jvp_v = grads[:, 2]      # ≈ -eps·a
        D_v_mean = jvp_lap_v.mean()
        D_v_std = jvp_lap_v.std()
        eps_mean = jvp_u.mean()
        eps_std = jvp_u.std()
        eps_a_mean = -jvp_v.mean()
        eps_a_std = jvp_v.std()

        # Residual = mixer_output − Σ partial_i · feature_i ≈ -eps·b.
        feats_det = features.detach()
        mixer_output = fast_model.forward_one(1, feats_det).detach()
        lin_combo = (
            jvp_lap_v * feats_det[:, 0]
            + jvp_u * feats_det[:, 1]
            + jvp_v * feats_det[:, 2]
        )
        residual = mixer_output - lin_combo
        eps_b_mean = -residual.mean()
        eps_b_std = residual.std()

        return {
            "D_v": {"mean": D_v_mean, "std": D_v_std},
            "eps": {"mean": eps_mean, "std": eps_std},
            "eps_a": {"mean": eps_a_mean, "std": eps_a_std},
            "eps_b": {"mean": eps_b_mean, "std": eps_b_std},
        }

    def auxiliary_losses(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Differentiable nMSE aux losses against ground-truth FHN coefficients.

        Mirrors extract_coefficients but keeps the autograd graph alive.
        `targets` is unused — aux targets come from task metadata.
        """
        del targets  # unused; aux targets come from self.{D_u,D_v,eps,a,b}
        if mixer_idx not in (0, 1):
            raise ValueError(
                f"FitzHugh-Nagumo has n_outputs=2; mixer_idx must be 0 or 1, got {mixer_idx}"
            )

        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(mixer_idx, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, n_cols)

        def _nmse(quantity: torch.Tensor, target_value: float) -> torch.Tensor:
            target_per_point = torch.full_like(quantity, target_value)
            diff = quantity - target_per_point
            mse = (diff * diff).mean()
            denom = (target_per_point * target_per_point).mean().clamp(min=1e-12)
            return mse / denom

        if mixer_idx == 0:
            jvp_lap_u = grads[:, 0]
            aux_D_u = _nmse(jvp_lap_u, self.D_u)
            return {"D_u": aux_D_u}

        # mixer_v (idx == 1); Framing A — compound coefficients
        jvp_lap_v = grads[:, 0]
        jvp_u = grads[:, 1]
        jvp_v = grads[:, 2]
        aux_D_v = _nmse(jvp_lap_v, self.D_v)
        aux_eps = _nmse(jvp_u, self.eps)
        aux_eps_a = _nmse(-jvp_v, self.eps * self.a)

        # Residual for eps_b — differentiable version uses features_grad.
        mixer_output = fast_model.forward_one(1, features_grad)
        lin_combo = (
            jvp_lap_v * features_grad[:, 0]
            + jvp_u * features_grad[:, 1]
            + jvp_v * features_grad[:, 2]
        )
        residual = mixer_output - lin_combo
        aux_eps_b = _nmse(-residual, self.eps * self.b)

        return {
            "D_v": aux_D_v,
            "eps": aux_eps,
            "eps_a": aux_eps_a,
            "eps_b": aux_eps_b,
        }

    def inject_noise_at_source(
        self,
        hat_tensors: dict[str, torch.Tensor],
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Real-space Gaussian noise via Fourier round-trip on u_hat and v_hat."""
        if noise_level <= 0.0:
            return hat_tensors
        u_hat = hat_tensors["u_hat"]
        v_hat = hat_tensors["v_hat"]

        u_real = torch.fft.ifft2(u_hat).real
        u_std = u_real.std()
        u_noise_real = torch.randn(
            u_real.shape,
            dtype=u_real.dtype,
            device=u_hat.device,
            generator=generator,
        ) * (noise_level * u_std)
        u_noise_hat = torch.fft.fft2(u_noise_real.to(dtype=u_hat.dtype))

        v_real = torch.fft.ifft2(v_hat).real
        v_std = v_real.std()
        v_noise_real = torch.randn(
            v_real.shape,
            dtype=v_real.dtype,
            device=v_hat.device,
            generator=generator,
        ) * (noise_level * v_std)
        v_noise_hat = torch.fft.fft2(v_noise_real.to(dtype=v_hat.dtype))

        return {"u_hat": u_hat + u_noise_hat, "v_hat": v_hat + v_noise_hat}


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
        self.u_hat = torch.tensor(data["u_hat"], dtype=torch.complex128, device=self.storage_device)
        self.v_hat = torch.tensor(data["v_hat"], dtype=torch.complex128, device=self.storage_device)

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

    def evaluate_collocations(
        self,
        snap_idx_list: torch.Tensor,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Produce structural features and targets for Lambda-Omega.

        Mixer_u library: [u_xx+u_yy, u, u³, u·v², u²·v, v³]
        Mixer_v library: [v_xx+v_yy, v, u³, u·v², u²·v, v³]
        Targets: analytical u_t, v_t from the expanded λ-ω RHS.
        """
        neg_kx2 = -((self.kx.unsqueeze(0)) ** 2)  # (1, nx)
        neg_ky2 = -((self.ky.unsqueeze(1)) ** 2)  # (ny, 1)

        u_hat = self.u_hat[snap_idx_list].to(device=self.device)
        v_hat = self.v_hat[snap_idx_list].to(device=self.device)

        coeff_batch = torch.stack(
            [
                u_hat,
                v_hat,
                neg_kx2 * u_hat,  # u_xx
                neg_ky2 * u_hat,  # u_yy
                neg_kx2 * v_hat,  # v_xx
                neg_ky2 * v_hat,  # v_yy
            ],
            dim=0,
        )  # (6, n_unique, ny, nx)

        u, v, u_xx, u_yy, v_xx, v_yy = fourier_eval_2d(
            coeff_batch, E_x, E_y, self.device
        )

        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy
        u3 = u * u * u
        u_v2 = u * v * v
        u2_v = u * u * v
        v3 = v * v * v

        mixer_u_features = torch.stack(
            [lap_u, u, u3, u_v2, u2_v, v3], dim=2
        )  # (n_unique, n_pts, 6)
        mixer_v_features = torch.stack(
            [lap_v, v, u3, u_v2, u2_v, v3], dim=2
        )  # (n_unique, n_pts, 6)

        # Expanded λ-ω RHS:
        #   u_t = D_u·∇²u + a·u - u³ - u·v² - c·u²·v - c·v³
        #   v_t = D_v·∇²v + a·v + c·u³ + c·u·v² - u²·v - v³
        u_t = (
            self.D_u * lap_u
            + self.a * u
            - u3
            - u_v2
            - self.c * u2_v
            - self.c * v3
        )
        v_t = (
            self.D_v * lap_v
            + self.a * v
            + self.c * u3
            + self.c * u_v2
            - u2_v
            - v3
        )
        targets = torch.stack([u_t, v_t], dim=2)  # (n_unique, n_pts, 2)

        return [mixer_u_features.float(), mixer_v_features.float()], targets.float()

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
    def rhs_feature_mask(self) -> list[bool]:
        # λ-ω RHS: D_u*∇²u + a*u - (u+c*v)*r², D_v*∇²v + a*v + (c*u-v)*r²
        # Uses: u, v, u_xx, u_yy, v_xx, v_yy. NOT: u_x, u_y, v_x, v_y
        return [True, True, False, False, True, True, False, False, True, True]

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

    # ── Mixer method implementations ─────────────────────────────────

    @property
    def n_outputs(self) -> int:
        return 2

    @property
    def structural_feature_names(self) -> list[list[str]]:
        return [
            ["u_xx+u_yy", "u", "u³", "u·v²", "u²·v", "v³"],
            ["v_xx+v_yy", "v", "u³", "u·v²", "u²·v", "v³"],
        ]

    @property
    def aux_loss_names(self) -> list[list[str]]:
        return [
            ["D_u", "a_from_u", "c_from_u2v", "c_from_v3"],
            ["D_v", "a_from_v", "c_from_u3", "c_from_uv2"],
        ]

    def extract_coefficients(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Recover λ-ω coefficients via JVP through each mixer.

        mixer_u (idx=0) → {D_u, a_from_u, c_from_u2v, c_from_v3}.
        mixer_v (idx=1) → {D_v, a_from_v, c_from_u3, c_from_uv2}.
        """
        if mixer_idx not in (0, 1):
            raise ValueError(
                f"Lambda-Omega has n_outputs=2; mixer_idx must be 0 or 1, got {mixer_idx}"
            )

        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(mixer_idx, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=False,
            retain_graph=False,
        )[0].detach()  # (N, 6)

        if mixer_idx == 0:
            # Library: [lap_u, u, u³, u·v², u²·v, v³]
            # Weights: (D_u, +a, -1, -1, -c, -c)
            jvp_lap_u = grads[:, 0]
            jvp_u = grads[:, 1]
            jvp_u2v = grads[:, 4]  # ≈ -c
            jvp_v3 = grads[:, 5]   # ≈ -c
            D_u_mean = jvp_lap_u.mean()
            D_u_std = jvp_lap_u.std()
            a_mean = jvp_u.mean()
            a_std = jvp_u.std()
            c_u2v_vals = -jvp_u2v
            c_u2v_mean = c_u2v_vals.mean()
            c_u2v_std = jvp_u2v.std()
            c_v3_vals = -jvp_v3
            c_v3_mean = c_v3_vals.mean()
            c_v3_std = jvp_v3.std()
            return {
                "D_u": {"mean": D_u_mean, "std": D_u_std},
                "a_from_u": {"mean": a_mean, "std": a_std},
                "c_from_u2v": {"mean": c_u2v_mean, "std": c_u2v_std},
                "c_from_v3": {"mean": c_v3_mean, "std": c_v3_std},
            }

        # mixer_v (idx == 1)
        # Library: [lap_v, v, u³, u·v², u²·v, v³]
        # Weights: (D_v, +a, +c, +c, -1, -1)
        jvp_lap_v = grads[:, 0]
        jvp_v = grads[:, 1]
        jvp_u3 = grads[:, 2]   # ≈ +c
        jvp_uv2 = grads[:, 3]  # ≈ +c
        D_v_mean = jvp_lap_v.mean()
        D_v_std = jvp_lap_v.std()
        a_mean = jvp_v.mean()
        a_std = jvp_v.std()
        c_u3_mean = jvp_u3.mean()
        c_u3_std = jvp_u3.std()
        c_uv2_mean = jvp_uv2.mean()
        c_uv2_std = jvp_uv2.std()
        return {
            "D_v": {"mean": D_v_mean, "std": D_v_std},
            "a_from_v": {"mean": a_mean, "std": a_std},
            "c_from_u3": {"mean": c_u3_mean, "std": c_u3_std},
            "c_from_uv2": {"mean": c_uv2_mean, "std": c_uv2_std},
        }

    def auxiliary_losses(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Differentiable nMSE aux losses against ground-truth λ-ω coefficients."""
        del targets  # unused; aux targets come from self.{D_u,D_v,a,c}
        if mixer_idx not in (0, 1):
            raise ValueError(
                f"Lambda-Omega has n_outputs=2; mixer_idx must be 0 or 1, got {mixer_idx}"
            )

        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(mixer_idx, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 6)

        def _nmse(quantity: torch.Tensor, target_value: float) -> torch.Tensor:
            target_per_point = torch.full_like(quantity, target_value)
            diff = quantity - target_per_point
            mse = (diff * diff).mean()
            denom = (target_per_point * target_per_point).mean().clamp(min=1e-12)
            return mse / denom

        if mixer_idx == 0:
            jvp_lap_u = grads[:, 0]
            jvp_u = grads[:, 1]
            jvp_u2v = grads[:, 4]
            jvp_v3 = grads[:, 5]
            aux_D_u = _nmse(jvp_lap_u, self.D_u)
            aux_a_from_u = _nmse(jvp_u, self.a)
            aux_c_from_u2v = _nmse(-jvp_u2v, self.c)
            aux_c_from_v3 = _nmse(-jvp_v3, self.c)
            return {
                "D_u": aux_D_u,
                "a_from_u": aux_a_from_u,
                "c_from_u2v": aux_c_from_u2v,
                "c_from_v3": aux_c_from_v3,
            }

        # mixer_v (idx == 1)
        jvp_lap_v = grads[:, 0]
        jvp_v = grads[:, 1]
        jvp_u3 = grads[:, 2]
        jvp_uv2 = grads[:, 3]
        aux_D_v = _nmse(jvp_lap_v, self.D_v)
        aux_a_from_v = _nmse(jvp_v, self.a)
        aux_c_from_u3 = _nmse(jvp_u3, self.c)
        aux_c_from_uv2 = _nmse(jvp_uv2, self.c)
        return {
            "D_v": aux_D_v,
            "a_from_v": aux_a_from_v,
            "c_from_u3": aux_c_from_u3,
            "c_from_uv2": aux_c_from_uv2,
        }

    def inject_noise_at_source(
        self,
        hat_tensors: dict[str, torch.Tensor],
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Real-space Gaussian noise via Fourier round-trip on u_hat and v_hat."""
        if noise_level <= 0.0:
            return hat_tensors
        u_hat = hat_tensors["u_hat"]
        v_hat = hat_tensors["v_hat"]

        u_real = torch.fft.ifft2(u_hat).real
        u_std = u_real.std()
        u_noise_real = torch.randn(
            u_real.shape,
            dtype=u_real.dtype,
            device=u_hat.device,
            generator=generator,
        ) * (noise_level * u_std)
        u_noise_hat = torch.fft.fft2(u_noise_real.to(dtype=u_hat.dtype))

        v_real = torch.fft.ifft2(v_hat).real
        v_std = v_real.std()
        v_noise_real = torch.randn(
            v_real.shape,
            dtype=v_real.dtype,
            device=v_hat.device,
            generator=generator,
        ) * (noise_level * v_std)
        v_noise_hat = torch.fft.fft2(v_noise_real.to(dtype=v_hat.dtype))

        return {"u_hat": u_hat + u_noise_hat, "v_hat": v_hat + v_noise_hat}


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
        self.u_hat = torch.tensor(data["u_hat"], dtype=torch.complex128, device=self.storage_device)
        self.v_hat = torch.tensor(data["v_hat"], dtype=torch.complex128, device=self.storage_device)
        self.p_hat = torch.tensor(data["p_hat"], dtype=torch.complex128, device=self.storage_device)

    def _extract_pde_params(self) -> None:
        self.nu = float(self.simulation_params.get("nu") or self.ic_config.get("nu"))
        if self.nu is None:
            raise Exception("NS coefficient nu is None")

    def evaluate_collocations(
        self,
        snap_idx_list: torch.Tensor,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Produce structural features and targets for NS via the vorticity form.

        Library: [u·ω_x + v·ω_y, ω_xx+ω_yy] for the single mixer.
        Target:  ω_t = -(u·ω_x + v·ω_y) + nu·(ω_xx+ω_yy).
        """
        ikx = 1j * self.kx.unsqueeze(0)  # (1, nx)
        iky = 1j * self.ky.unsqueeze(1)  # (ny, 1)
        neg_kx2 = -((self.kx.unsqueeze(0)) ** 2)  # (1, nx)
        neg_ky2 = -((self.ky.unsqueeze(1)) ** 2)  # (ny, 1)
        neg_kx2_plus_ky2 = neg_kx2 + neg_ky2  # (ny, nx)

        u_hat = self.u_hat[snap_idx_list].to(device=self.device)
        v_hat = self.v_hat[snap_idx_list].to(device=self.device)

        omega_hat = ikx * v_hat - iky * u_hat
        omega_x_hat = ikx * omega_hat
        omega_y_hat = iky * omega_hat
        lap_omega_hat = neg_kx2_plus_ky2 * omega_hat

        coeff_batch = torch.stack(
            [
                u_hat,
                v_hat,
                omega_x_hat,
                omega_y_hat,
                lap_omega_hat,
            ],
            dim=0,
        )  # (5, n_unique, ny, nx)

        u, v, omega_x, omega_y, lap_omega = fourier_eval_2d(
            coeff_batch, E_x, E_y, self.device
        )

        advection = u * omega_x + v * omega_y  # (n_unique, n_pts)

        mixer_0_features = torch.stack(
            [advection, lap_omega], dim=2
        )  # (n_unique, n_pts, 2)

        omega_t = -advection + self.nu * lap_omega
        targets = torch.stack([omega_t], dim=2)  # (n_unique, n_pts, 1)

        return [mixer_0_features.float()], targets.float()

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
    def rhs_feature_mask(self) -> list[bool]:
        # NS RHS: -(u*u_x + v*u_y) - p_x + nu*∇²u, -(u*v_x + v*v_y) - p_y + nu*∇²v
        # ALL 10 features appear in the RHS (advection uses first derivatives)
        return [True, True, True, True, True, True, True, True, True, True]

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

    # ── Mixer method implementations ─────────────────────────────────

    @property
    def n_outputs(self) -> int:
        return 1

    @property
    def structural_feature_names(self) -> list[list[str]]:
        return [["u·ω_x+v·ω_y", "ω_xx+ω_yy"]]

    @property
    def aux_loss_names(self) -> list[list[str]]:
        return [["nu"]]

    def extract_coefficients(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Recover nu via the direct partial wrt the vorticity Laplacian column.

        The mixer ideally learns `f(a, b) ≈ -a + nu·b` where a is the
        advection composite and b = ω_xx+ω_yy. Then ∂f/∂b = nu pointwise.
        """
        if mixer_idx != 0:
            raise ValueError(
                f"NS has n_outputs=1; mixer_idx must be 0, got {mixer_idx}"
            )
        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(0, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=False,
            retain_graph=False,
        )[0]  # (N, 2)
        jvp_lap_omega = grads[:, 1].detach()
        nu_mean = jvp_lap_omega.mean()
        nu_std = jvp_lap_omega.std()
        return {"nu": {"mean": nu_mean, "std": nu_std}}

    def auxiliary_losses(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Form 1 per-point JVP target for nu.

        Pushes ∂(mixer)/∂(ω_xx+ω_yy) toward task.nu at every collocation
        point. Differentiable via create_graph=True on the autograd.grad call.
        """
        if mixer_idx != 0:
            raise ValueError(
                f"NS has n_outputs=1; mixer_idx must be 0, got {mixer_idx}"
            )
        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(0, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 2)
        jvp_lap_omega = grads[:, 1]  # (N,)
        target_per_point = torch.full_like(jvp_lap_omega, float(self.nu))
        diff = jvp_lap_omega - target_per_point
        mse = (diff * diff).mean()
        denom = (target_per_point * target_per_point).mean().clamp(min=1e-12)
        aux_nu = mse / denom
        return {"nu": aux_nu}

    def inject_noise_at_source(
        self,
        hat_tensors: dict[str, torch.Tensor],
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Inject real-space Gaussian noise into u and v via the Fourier round-trip.

        p_hat is legacy under the vorticity formulation and is passed through
        unchanged if present.
        """
        if noise_level <= 0.0:
            return hat_tensors
        out: dict[str, torch.Tensor] = {}
        for key in ("u_hat", "v_hat"):
            if key not in hat_tensors:
                continue
            hat = hat_tensors[key]
            real = torch.fft.ifft2(hat).real
            std = real.std()
            noise_real = torch.randn(
                real.shape,
                dtype=real.dtype,
                device=hat.device,
                generator=generator,
            ) * (noise_level * std)
            noise_hat = torch.fft.fft2(noise_real.to(dtype=hat.dtype))
            out[key] = hat + noise_hat
        for k, v in hat_tensors.items():
            if k not in out:
                out[k] = v
        return out


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
        self.u_hat = torch.tensor(data["u_hat"], dtype=torch.complex128, device=self.storage_device)

    def _extract_pde_params(self) -> None:
        D = self.simulation_params.get("D")
        if D is None:
            D = self.ic_config.get("D_used")
        if D is None:
            raise ValueError(
                "Heat equation coefficient D missing from simulation_params and ic_config"
            )
        self.D = float(D)

    def evaluate_collocations(
        self,
        snap_idx_list: torch.Tensor,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Produce structural features and targets for Heat.

        Library: [u_xx, u_yy] for the single mixer — two primitives
        rather than the pre-summed Laplacian. The mixer learns that both
        should have weight D, giving two recovery paths for D.
        Target:  u_t = D * (u_xx + u_yy) (analytical).
        """
        neg_kx2 = -((self.kx.unsqueeze(0)) ** 2)  # (1, nx)
        neg_ky2 = -((self.ky.unsqueeze(1)) ** 2)  # (ny, 1)

        u_hat = self.u_hat[snap_idx_list].to(device=self.device)

        coeff_batch = torch.stack(
            [
                neg_kx2 * u_hat,  # u_xx
                neg_ky2 * u_hat,  # u_yy
            ],
            dim=0,
        )  # (2, n_unique, ny, nx)

        u_xx, u_yy = fourier_eval_2d(coeff_batch, E_x, E_y, self.device)

        # Structural features for the single Heat mixer: [u_xx, u_yy]
        mixer_0_features = torch.stack([u_xx, u_yy], dim=2)  # (n_unique, n_pts, 2)

        u_t = self.D * (u_xx + u_yy)
        targets = torch.stack([u_t], dim=2)  # (n_unique, n_pts, 1)

        return [mixer_0_features.float()], targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy post-feature noise injection (scheduled for deletion in M5)."""
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + noise
        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"D": self.D}

    # ── Mixer method implementations ─────────────────────────────────

    @property
    def n_outputs(self) -> int:
        return 1

    @property
    def structural_feature_names(self) -> list[list[str]]:
        return [["u_xx", "u_yy"]]

    @property
    def aux_loss_names(self) -> list[list[str]]:
        return [["D_from_uxx", "D_from_uyy"]]

    def extract_coefficients(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Recover D from both feature columns (two independent paths).

        The mixer ideally learns `f(a, b) ≈ D·(a + b)`. Both partials
        ∂f/∂a and ∂f/∂b should equal D. We report both means separately
        as `D_from_uxx` and `D_from_uyy` — they serve as a cross-check
        on whether the mixer learned symmetric weights on the two
        Laplacian components.
        """
        if mixer_idx != 0:
            raise ValueError(
                f"Heat has n_outputs=1; mixer_idx must be 0, got {mixer_idx}"
            )
        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(0, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=False,
            retain_graph=False,
        )[0]  # (N, 2)
        jvp_uxx = grads[:, 0].detach()  # ≈ D per point
        jvp_uyy = grads[:, 1].detach()  # ≈ D per point
        return {
            "D_from_uxx": {"mean": jvp_uxx.mean().detach(), "std": jvp_uxx.std().detach()},
            "D_from_uyy": {"mean": jvp_uyy.mean().detach(), "std": jvp_uyy.std().detach()},
        }

    def auxiliary_losses(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Push both ∂f/∂u_xx and ∂f/∂u_yy toward D at every point.

        Two aux losses (one per recovery path) — enforces that the
        mixer learns symmetric weights on the Laplacian components.
        """
        if mixer_idx != 0:
            raise ValueError(
                f"Heat has n_outputs=1; mixer_idx must be 0, got {mixer_idx}"
            )
        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(0, features_grad)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 2)
        jvp_uxx = grads[:, 0]
        jvp_uyy = grads[:, 1]
        target_per_point = torch.full_like(jvp_uxx, self.D)
        denom = (target_per_point * target_per_point).mean().clamp(min=1e-12)
        diff_uxx = jvp_uxx - target_per_point
        diff_uyy = jvp_uyy - target_per_point
        aux_D_uxx = (diff_uxx * diff_uxx).mean() / denom
        aux_D_uyy = (diff_uyy * diff_uyy).mean() / denom
        return {"D_from_uxx": aux_D_uxx, "D_from_uyy": aux_D_uyy}

    def inject_noise_at_source(
        self,
        hat_tensors: dict[str, torch.Tensor],
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Real-space Gaussian noise via Fourier round-trip (same pattern as NLHeat)."""
        if noise_level <= 0.0:
            return hat_tensors
        u_hat = hat_tensors["u_hat"]
        u_real = torch.fft.ifft2(u_hat).real
        u_std = u_real.std()
        noise_real = torch.randn(
            u_real.shape,
            dtype=u_real.dtype,
            device=u_hat.device,
            generator=generator,
        ) * (noise_level * u_std)
        noise_hat = torch.fft.fft2(noise_real.to(dtype=u_hat.dtype))
        return {"u_hat": u_hat + noise_hat}

    @property
    def rhs_feature_mask(self) -> list[bool]:
        # Heat RHS: D * (u_xx + u_yy)
        # Uses: u_xx, u_yy. NOT: u, u_x, u_y
        return [False, False, False, True, True]

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
    jacobian_plot_type: str = "scatter"

    def _load_coefficients(self, data: np.lib.npyio.NpzFile) -> None:
        self.n_snapshots = data["u_hat"].shape[0]
        self.ny, self.nx = data["u_hat"].shape[1], data["u_hat"].shape[2]
        self.u_hat = torch.tensor(data["u_hat"], dtype=torch.complex128, device=self.storage_device)

    def _extract_pde_params(self) -> None:
        K = self.simulation_params.get("K")
        if K is None:
            K = self.ic_config.get("K_used")
        if K is None:
            raise ValueError(
                "Nonlinear heat coefficient K missing from simulation_params and ic_config"
            )
        self.K = float(K)

    def evaluate_collocations(
        self,
        snap_idx_list: torch.Tensor,
        E_x: torch.Tensor,
        E_y: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Produce structural features and targets for NLHeat.

        Library: [1-u, u_xx+u_yy] for the single mixer.
        Target:  u_t = K * (1-u) * (u_xx+u_yy) (analytical, from self.K).
        """
        neg_kx2 = -((self.kx.unsqueeze(0)) ** 2)  # (1, nx)
        neg_ky2 = -((self.ky.unsqueeze(1)) ** 2)  # (ny, 1)

        u_hat = self.u_hat[snap_idx_list].to(device=self.device)

        coeff_batch = torch.stack(
            [
                u_hat,
                neg_kx2 * u_hat,  # u_xx
                neg_ky2 * u_hat,  # u_yy
            ],
            dim=0,
        )  # (3, n_unique, ny, nx)

        u, u_xx, u_yy = fourier_eval_2d(coeff_batch, E_x, E_y, self.device)

        lap_u = u_xx + u_yy
        one_minus_u = 1.0 - u

        # Structural features for the single NLHeat mixer: [1-u, u_xx+u_yy]
        mixer_0_features = torch.stack(
            [one_minus_u, lap_u], dim=2
        )  # (n_unique, n_pts, 2)

        u_t = self.K * one_minus_u * lap_u
        targets = torch.stack([u_t], dim=2)  # (n_unique, n_pts, 1)

        return [mixer_0_features.float()], targets.float()

    def inject_noise(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy post-feature noise injection (scheduled for deletion).

        Still required by PDETask abstract interface; will be removed
        when the source-level noise hook is wired into the training
        loop in Milestone 5.
        """
        feat_std = features.std(dim=0, keepdim=True)
        noise = torch.randn(
            features.shape,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        ) * (noise_level * feat_std)
        features = features + noise
        return features, targets

    @property
    def diffusion_coeffs(self) -> dict:
        return {"K": self.K}

    @property
    def rhs_feature_mask(self) -> list[bool]:
        # Legacy (pre-structural) mask, 5-column raw layout.
        # Kept for documentation only.
        return [True, False, False, True, True]

    @staticmethod
    def _extract_K(jvp_per_point: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Legacy K recovery via least-squares (pre-structural API).

        Kept for backward compat with existing coefficient_specs path.
        Superseded by extract_coefficients under the mixer method.
        """
        one_minus_u = 1.0 - features[:, 0]
        K = np.dot(jvp_per_point, one_minus_u) / np.dot(one_minus_u, one_minus_u)
        return np.full_like(jvp_per_point, K)

    @property
    def coefficient_specs(self) -> list[CoefficientSpec]:
        return [
            CoefficientSpec(
                name="K", perturb_indices=[3, 4], output_index=0, true_value=self.K,
                post_extract=self._extract_K,
            ),
        ]

    # ── Mixer method implementations ─────────────────────────────────

    @property
    def n_outputs(self) -> int:
        return 1

    @property
    def structural_feature_names(self) -> list[list[str]]:
        return [["1-u", "u_xx+u_yy"]]

    @property
    def aux_loss_names(self) -> list[list[str]]:
        return [["K"]]

    def extract_coefficients(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Recover K via regression slope on the mixer's per-point partials.

        The mixer ideally learns `f(a, b) ≈ K·a·b` where a = 1-u and
        b = u_xx+u_yy. The partial ∂f/∂a equals K·b at every point. A
        least-squares slope of (∂f/∂a) against b recovers K.
        """
        if mixer_idx != 0:
            raise ValueError(
                f"NLHeat has n_outputs=1; mixer_idx must be 0, got {mixer_idx}"
            )
        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(0, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=False,
            retain_graph=False,
        )[0]  # (N, 2)
        jvp_1_minus_u = grads[:, 0].detach()  # ≈ K * (u_xx+u_yy) per point
        lap = features[:, 1].detach()  # (u_xx+u_yy) per point
        numer = (jvp_1_minus_u * lap).sum()
        denom = (lap * lap).sum().clamp(min=1e-12)
        K_slope = numer / denom
        K_std = jvp_1_minus_u.std()  # per-point dispersion (linearity check)
        return {"K": {"mean": K_slope.detach(), "std": K_std.detach()}}

    def auxiliary_losses(
        self,
        mixer_idx: int,
        fast_model,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Form 1 per-point JVP target for K.

        Pushes ∂(mixer)/∂(1-u) toward K·(u_xx+u_yy) at every collocation
        point. Differentiable — contributes gradient to the mixer's
        weights via create_graph=True on the inner autograd.grad call.
        """
        if mixer_idx != 0:
            raise ValueError(
                f"NLHeat has n_outputs=1; mixer_idx must be 0, got {mixer_idx}"
            )
        features_grad = features.detach().clone().requires_grad_(True)
        output = fast_model.forward_one(0, features_grad)  # (N,)
        grads = torch.autograd.grad(
            output.sum(),
            features_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N, 2)
        jvp_1_minus_u = grads[:, 0]  # (N,)
        lap = features[:, 1]
        target_per_point = self.K * lap
        diff = jvp_1_minus_u - target_per_point
        mse = (diff * diff).mean()
        denom = (target_per_point * target_per_point).mean().clamp(min=1e-12)
        aux_K = mse / denom
        return {"K": aux_K}

    def inject_noise_at_source(
        self,
        hat_tensors: dict[str, torch.Tensor],
        noise_level: float,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Inject real-space Gaussian noise into u via the Fourier round-trip.

        Generates white noise in real space scaled to noise_level * std(u),
        FFTs it, and adds to u_hat. The result is that derivatives computed
        from the noisy u_hat inherit correlated noise through the spectral
        operators — matching the way real-world noise would propagate.
        """
        if noise_level <= 0.0:
            return hat_tensors
        u_hat = hat_tensors["u_hat"]
        u_real = torch.fft.ifft2(u_hat).real  # (*, ny, nx)
        u_std = u_real.std()
        noise_real = torch.randn(
            u_real.shape,
            dtype=u_real.dtype,
            device=u_hat.device,
            generator=generator,
        ) * (noise_level * u_std)
        noise_hat = torch.fft.fft2(noise_real.to(dtype=u_hat.dtype))
        return {"u_hat": u_hat + noise_hat}


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

        # Auto-promote *_hat to GPU if enough VRAM
        if device == "cuda" and torch.cuda.is_available() and len(self.tasks) > 0:
            total_hat = sum(t.hat_memory_bytes() for t in self.tasks)
            free_vram = torch.cuda.mem_get_info()[0]
            headroom = 3 * (1024 ** 3)  # reserve 3 GB for model + collocation transients
            if total_hat < (free_vram - headroom):
                for t in self.tasks:
                    t.promote_storage("cuda")
                print(f"\n  [task_loader] Promoted storage tensors to GPU ({total_hat / 1024**3:.1f} GB, "
                      f"free: {free_vram / 1024**3:.1f} GB)")
            else:
                print(f"\n  [task_loader] Keeping storage tensors on CPU ({total_hat / 1024**3:.1f} GB, "
                      f"free: {free_vram / 1024**3:.1f} GB)")

    def sample_batch(self, n_tasks: int, seed: Optional[int] = None) -> List[PDETask]:
        """
        Sample random subset of tasks for meta-training batch.
        """
        if n_tasks >= len(self.tasks):
            return list(self.tasks)

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
