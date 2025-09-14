"""
Stratified Batch Sampler for AUC Loss
------------------------------------
Ensures each batch contains a balanced representation of classes.
This is critical for AUC-based losses, which rely on pairwise comparisons
between positive and negative examples.

Author: Khashayar (Ernest) Namdar
GitHub: https://github.com/knamdar
License: MIT
"""

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, DataLoader


class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampler for classification tasks.

    Ensures that every batch contains at least one sample from each class,
    which is important for pairwise ranking losses (e.g., AUC loss).

    Args:
        labels (np.ndarray or Tensor): Array of class labels (shape: [N]).
        batch_size (int): Number of samples per batch.

    Example:
        >>> labels = np.array([0, 0, 1, 1, 1])
        >>> sampler = StratifiedBatchSampler(labels, batch_size=4)
        >>> for batch in sampler:
        ...     print(batch)  # Each batch contains both 0s and 1s
    """

    def __init__(self, labels, batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = len(np.unique(self.labels))
        self.class_indices = {
            cls: np.where(self.labels == cls)[0].tolist()
            for cls in np.unique(self.labels)
        }
        self.num_samples = len(self.labels)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)

        used_samples = np.zeros(self.num_samples, dtype=bool)
        batches = []

        while not np.all(used_samples):
            batch = []
            # Ensure at least one sample per class
            for cls in range(self.num_classes):
                cls_indices = [idx for idx in self.class_indices[cls] if not used_samples[idx]]
                if cls_indices:
                    chosen_idx = np.random.choice(cls_indices, 1, replace=False).tolist()
                    used_samples[chosen_idx] = True
                else:
                    # Oversample class if exhausted
                    chosen_idx = np.random.choice(self.class_indices[cls], 1, replace=True).tolist()
                batch.extend(chosen_idx)

            # Fill the rest of the batch randomly
            remaining_size = self.batch_size - len(batch)
            if remaining_size > 0:
                unused_indices = [i for i in indices if not used_samples[i]]
                if unused_indices:
                    chosen_indices = np.random.choice(
                        unused_indices, min(remaining_size, len(unused_indices)),
                        replace=False
                    ).tolist()
                    used_samples[chosen_indices] = True
                else:
                    # If nothing left, sample with replacement
                    chosen_indices = np.random.choice(indices, remaining_size, replace=True).tolist()
                batch.extend(chosen_indices)

            np.random.shuffle(batch)
            batches.append(batch)

        return iter(batches)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class ImbalancedRandomDataset(Dataset):
    """
    Utility dataset for testing stratified samplers.

    Generates a synthetic dataset with controlled class imbalance,
    useful for verifying that the sampler creates well-balanced batches.
    """

    def __init__(self, num_samples, num_features, num_classes, imbalance_ratios):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, num_features).astype(np.float32)
        self.pIDs = np.arange(num_samples)  # Patient IDs

        class_counts = [int(num_samples * ratio) for ratio in imbalance_ratios]
        remaining_samples = num_samples - sum(class_counts)
        class_counts[0] += remaining_samples  # Adjust for rounding errors

        self.labels = np.concatenate([
            np.full(count, cls, dtype=np.int64)
            for cls, count in enumerate(class_counts)
        ])

        # Shuffle data and labels together
        perm = np.random.permutation(num_samples)
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.pIDs[index]


def create_dataloader(num_samples, num_features, num_classes, batch_size, imbalance_ratios):
    dataset = ImbalancedRandomDataset(num_samples, num_features, num_classes, imbalance_ratios)
    sampler = StratifiedBatchSampler(dataset.labels, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    labels, counts = np.unique(dataset.labels, return_counts=True)
    print(f"Class distribution: {dict(zip(labels, counts))}")
    return dataloader


def demo():
    """
    Demonstration of the sampler with multiple batch sizes and class configurations.
    """
    batch_sizes = [2, 3, 4, 5]
    num_samples = 31
    num_features = 5
    imbalance_ratios = [
        [0.7, 0.3],        # Binary classification
        [0.6, 0.25, 0.15], # 3-class classification
        [0.5, 0.2, 0.2, 0.1] # 4-class classification
    ]

    for num_classes, ratios in zip([2, 3, 4], imbalance_ratios):
        print(f"\n--- {num_classes}-Class Classification ---")
        for batch_size in batch_sizes:
            if batch_size < num_classes:
                print(f"Skipping batch size {batch_size} (too small for {num_classes} classes)")
                continue

            print(f"\nBatch Size: {batch_size}")
            dataloader = create_dataloader(num_samples, num_features, num_classes, batch_size, ratios)
            for batch_idx, (data, labels, pIDs) in enumerate(dataloader):
                print(f"Batch {batch_idx + 1}")
                print(f"Patient IDs: {pIDs.numpy()}")
                print(f"Labels: {labels.numpy()}")


if __name__ == "__main__":
    demo()
