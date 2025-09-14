"""
AUC Loss Package

Author: Khashayar (Ernest) Namdar
GitHub: https://github.com/knamdar

Provides differentiable AUC-based loss functions and a stratified batch sampler
for PyTorch training workflows.

Usage:
    >>> from auc_loss import AUCLoss, MulticlassAUCLoss
"""


from .auc_loss import AUCLoss, MulticlassAUCLoss
from .sampler import StratifiedBatchSampler

__all__ = ["AUCLoss", "MulticlassAUCLoss", "StratifiedBatchSampler"]
