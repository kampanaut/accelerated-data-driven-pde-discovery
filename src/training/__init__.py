"""
Training infrastructure for meta-learning on PDE discovery.
"""

from .task_loader import (
    PDETask,
    BrusselatorTask,
    FitzHughNagumoTask,
    LambdaOmegaTask,
    NavierStokesTask,
    MetaLearningDataLoader,
)
from .maml import (
    MAMLConfig,
    MAMLTrainer,
    fine_tune,
    get_meta_learned_init,
)

__all__ = [
    "PDETask",
    "BrusselatorTask",
    "FitzHughNagumoTask",
    "LambdaOmegaTask",
    "NavierStokesTask",
    "MetaLearningDataLoader",
    "MAMLConfig",
    "MAMLTrainer",
    "fine_tune",
    "get_meta_learned_init",
]
