"""
AUC Loss Functions for PyTorch
------------------------------
Author: Khashayar (Ernest) Namdar
GitHub: https://github.com/knamdar
License: MIT
"""

import torch
from torch import nn, Tensor

# Shared softmax layer (applied along class dimension)
_softmax = nn.Softmax(dim=1)

def param_sigmoid(x: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Parameterized sigmoid function.

    Args:
        x (Tensor): Input tensor.
        alpha (float): Controls steepness of the sigmoid curve.
                       Higher alpha -> steeper transition.

    Returns:
        Tensor: Output after applying the sigmoid transformation.
    """
    return 1.0 / (1.0 + torch.exp(-alpha * x))


class AUCLoss(nn.Module):
    """
    Differentiable AUC-based ranking loss for binary classification.

    Encourages predicted probabilities for positive samples to be
    higher than those for negative samples using a differentiable
    pairwise ranking formulation.
    """

    def __init__(self, alpha: float = 20.0):
        """
        Args:
            alpha (float): Steepness parameter for the sigmoid.
                           Default = 20.0
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Compute the AUC loss.

        Args:
            output (Tensor): Model outputs of shape (N, 2). Raw logits allowed.
            target (Tensor): Ground-truth labels of shape (N,), values in {0, 1}.

        Returns:
            Tensor: Scalar loss value.
        """
        probs = _softmax(output)[:, 1]
        pos_pred = probs[target == 1]
        neg_pred = probs[target == 0]

        if pos_pred.numel() == 0 or neg_pred.numel() == 0:
            return torch.tensor(0.0, device=output.device)

        pairwise_diff = pos_pred.unsqueeze(1) - neg_pred
        transformed = param_sigmoid(pairwise_diff, self.alpha)

        return 1.0 - transformed.mean()


class MulticlassAUCLoss(nn.Module):
    """
    Differentiable multi-class AUC-based ranking loss.

    Computes one-vs-rest AUC per class and averages them.
    """

    def __init__(self, alpha: float = 20.0):
        """
        Args:
            alpha (float): Steepness parameter for the sigmoid.
                           Default = 20.0
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Compute the multiclass AUC loss.

        Args:
            output (Tensor): Model outputs of shape (N, C), where C is number of classes.
            target (Tensor): Ground-truth labels of shape (N,), values in [0, C-1].

        Returns:
            Tensor: Scalar loss value.
        """
        probs = _softmax(output)
        num_classes = probs.size(1)
        total_auc = 0.0
        valid_class_count = 0

        for class_idx in range(num_classes):
            pos_pred = probs[target == class_idx, class_idx]
            neg_pred = probs[target != class_idx, class_idx]

            if pos_pred.numel() == 0 or neg_pred.numel() == 0:
                continue  # Skip classes with no positives or no negatives

            pairwise_diff = pos_pred.unsqueeze(1) - neg_pred.unsqueeze(0)
            transformed = param_sigmoid(pairwise_diff, self.alpha)

            total_auc += transformed.mean()
            valid_class_count += 1

        if valid_class_count == 0:
            return torch.tensor(0.0, device=output.device)

        mean_auc = total_auc / valid_class_count
        return 1.0 - mean_auc
