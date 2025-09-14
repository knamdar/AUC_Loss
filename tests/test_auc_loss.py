import pytest
import torch
from auc_loss import AUCLoss, MulticlassAUCLoss


def test_auc_loss_binary_forward_and_backward():
    """Test that AUCLoss computes a scalar and supports backprop."""
    criterion = AUCLoss(alpha=10.0)

    # Create dummy logits: higher values for positives
    logits = torch.tensor([[1.5, 3.0],
                           [2.0, 4.0],
                           [3.0, 0.5],
                           [4.0, 1.0]], requires_grad=True)
    targets = torch.tensor([1, 1, 0, 0])

    loss = criterion(logits, targets)
    assert loss.ndim == 0, "Loss must be a scalar"
    assert 0.0 <= loss.detach().item() <= 1.0, "Loss must be between 0 and 1"

    # Backprop should work
    loss.backward()
    assert logits.grad is not None, "Gradient should be computed"


def test_auc_loss_binary_all_same_class_returns_zero():
    """If all samples are positives or all negatives, loss should be 0 (undefined case)."""
    criterion = AUCLoss(alpha=10.0)

    logits = torch.randn(4, 2, requires_grad=True)
    targets_all_pos = torch.ones(4, dtype=torch.long)

    loss = criterion(logits, targets_all_pos)
    assert loss.item() == 0.0


def test_multiclass_auc_loss_forward_and_backward():
    """Test that MulticlassAUCLoss computes a scalar and supports backprop."""
    criterion = MulticlassAUCLoss(alpha=10.0)

    # Create dummy logits for 3-class classification
    logits = torch.tensor([[3.0, 1.0, 0.5],
                           [2.5, 0.2, 1.5],
                           [0.5, 3.0, 1.0],
                           [0.1, 2.0, 3.0]], requires_grad=True)
    targets = torch.tensor([0, 0, 1, 2])

    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert 0.0 <= loss.detach().item() <= 1.0

    loss.backward()
    assert logits.grad is not None


def test_multiclass_auc_loss_skips_empty_classes():
    """Loss should skip classes with no positives or negatives and still return a scalar."""
    criterion = MulticlassAUCLoss(alpha=10.0)

    logits = torch.tensor([[3.0, 1.0, 0.5],
                           [2.5, 0.2, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 0])  # Only class 0 present → class 1 & 2 empty

    loss = criterion(logits, targets)
    assert loss.item() == 0.0 or 0.0 <= loss.item() <= 1.0
