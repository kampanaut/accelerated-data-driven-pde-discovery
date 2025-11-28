"""
Training infrastructure for meta-learning on PDE discovery.
"""

from .task_loader import NavierStokesTask, MetaLearningDataLoader
from .maml import (
    MAMLConfig,
    MAMLTrainer,
    fine_tune,
    get_meta_learned_init,
    compare_initializations,
)

__all__ = [
    'NavierStokesTask',
    'MetaLearningDataLoader',
    'MAMLConfig',
    'MAMLTrainer',
    'fine_tune',
    'get_meta_learned_init',
    'compare_initializations',
]
