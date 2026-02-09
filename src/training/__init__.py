"""
Training infrastructure for meta-learning on PDE discovery.
"""

from .task_loader import PDETask, BrusselatorTask, NavierStokesTask, MetaLearningDataLoader
from .maml import (
    MAMLConfig,
    MAMLTrainer,
    fine_tune,
    get_meta_learned_init,
)

__all__ = [
    "PDETask",
    "BrusselatorTask",
    "NavierStokesTask",
    "MetaLearningDataLoader",
    "MAMLConfig",
    "MAMLTrainer",
    "fine_tune",
    "get_meta_learned_init",
]
