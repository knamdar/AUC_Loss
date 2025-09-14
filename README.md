# AUC Loss (PyTorch)

Differentiable AUC-based loss functions and stratified batch sampling utilities for PyTorch.

This repository provides:
- **`AUCLoss`** – Binary classification AUC loss using a differentiable pairwise ranking formulation.
- **`MulticlassAUCLoss`** – Extension to multi-class classification using one-vs-rest formulation.
- **`StratifiedBatchSampler`** – Ensures each batch contains at least one positive and one negative sample (or one instance per class for multi-class tasks). This is crucial for pairwise ranking losses such as AUC.

---

## 📖 Background

This work was developed as part of my **Ph.D. thesis** at the **University of Toronto**, where I focused on building robust AI pipelines for medical image analysis and outcome prediction. AUC is a popular metric in medicine, and this repository offers a differentiable surrogate loss that allows direct optimization of AUC during training.

---

## 🚀 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/knamdar/AUC_Loss.git
cd AUC_Loss
pip install -e .
```

Requires **Python ≥3.8**, **PyTorch ≥1.10**, and **NumPy ≥1.20**.

---

## 🧠 Usage

### Binary Classification

```python
import torch
from auc_loss import AUCLoss

criterion = AUCLoss(alpha=20.0)
logits = torch.randn(8, 2, requires_grad=True)  # [batch, 2]
targets = torch.randint(0, 2, (8,))

loss = criterion(logits, targets)
loss.backward()
print("Binary AUC Loss:", loss.item())
```

### Multiclass Classification

```python
from auc_loss import MulticlassAUCLoss

criterion = MulticlassAUCLoss(alpha=20.0)
logits = torch.randn(8, 4, requires_grad=True)  # [batch, num_classes]
targets = torch.randint(0, 4, (8,))

loss = criterion(logits, targets)
loss.backward()
print("Multiclass AUC Loss:", loss.item())
```

### Using the Stratified Batch Sampler

```python
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from auc_loss import StratifiedBatchSampler

# Synthetic dataset
X = torch.randn(20, 10)
y = np.array([0]*10 + [1]*10)
dataset = TensorDataset(X, torch.tensor(y))

sampler = StratifiedBatchSampler(y, batch_size=6)
loader = DataLoader(dataset, batch_sampler=sampler)

for batch_data, batch_labels in loader:
    print("Batch labels:", batch_labels.numpy())  # Each batch has positives and negatives
```

---

## 🧪 Testing

Run unit tests with:

```bash
pytest tests/ -v
```

---

## 📚 Citation

If you use this work in your research, please cite:

> **Khashayar Namdar**, Matthias W. Wagner, Cynthia Hawkins, Uri Tabori, Birgit B. Ertl-Wagner, Farzad Khalvati.  
> *Improving Pediatric Low-Grade Neuroepithelial Tumors Molecular Subtype Identification Using a Novel AUROC Loss Function for Convolutional Neural Networks.*  
> [arXiv:2402.03547](https://arxiv.org/abs/2402.03547)

BibTeX:

```bibtex
@article{namdar2024auroc,
  title={Improving Pediatric Low-Grade Neuroepithelial Tumors Molecular Subtype Identification Using a Novel AUROC Loss Function for Convolutional Neural Networks},
  author={Namdar, Khashayar and Wagner, Matthias W. and Hawkins, Cynthia and Tabori, Uri and Ertl-Wagner, Birgit B. and Khalvati, Farzad},
  journal={arXiv preprint arXiv:2402.03547},
  year={2024},
  url={https://arxiv.org/abs/2402.03547}
}
```

---

## 📜 License

This repository is released under the **MIT License**.
