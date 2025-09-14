import numpy as np
import torch
from auc_loss import StratifiedBatchSampler


def test_stratified_batch_sampler_has_all_classes():
    """Each batch must contain at least one sample from each class."""
    labels = np.array([0] * 8 + [1] * 4)  # Imbalanced binary dataset
    batch_size = 4
    sampler = StratifiedBatchSampler(labels, batch_size=batch_size)

    for batch in sampler:
        batch_labels = labels[batch]
        unique_classes = np.unique(batch_labels)
        assert 0 in unique_classes and 1 in unique_classes, \
            f"Batch {batch} is missing one class!"


def test_sampler_len_matches_expected_batches():
    """Sampler __len__ must match number of batches generated."""
    labels = np.array([0] * 10 + [1] * 6)
    batch_size = 4
    sampler = StratifiedBatchSampler(labels, batch_size=batch_size)

    batches = list(iter(sampler))
    expected_batches = (len(labels) + batch_size - 1) // batch_size
    assert len(batches) == expected_batches
