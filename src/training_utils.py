from __future__ import annotations

import random

import numpy as np
import torch


def ensure_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    if logits.numel() == 0 or targets.numel() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "positive_rate": 0.0}
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    tp = float(((preds == 1) & (targets == 1)).sum().item())
    tn = float(((preds == 0) & (targets == 0)).sum().item())
    fp = float(((preds == 1) & (targets == 0)).sum().item())
    fn = float(((preds == 0) & (targets == 1)).sum().item())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(targets.mean().item()),
    }
