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
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
            "auprc": 0.0,
            "positive_rate": 0.0,
        }

    probs = torch.sigmoid(logits).detach().cpu()
    preds = (probs >= threshold).float()
    targets = targets.float().detach().cpu()

    tp = float(((preds == 1) & (targets == 1)).sum().item())
    tn = float(((preds == 0) & (targets == 0)).sum().item())
    fp = float(((preds == 1) & (targets == 0)).sum().item())
    fn = float(((preds == 0) & (targets == 1)).sum().item())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    auroc = 0.0
    auprc = 0.0

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        y_true = targets.numpy()
        y_prob = probs.numpy()

        if len(set(y_true.tolist())) >= 2:
            auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    except Exception:
        pass

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "positive_rate": float(targets.mean().item()),
    }

def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits: [B, M]
    targets: [B, M]
    mask: [B, M]
    """
    criterion = torch.nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=pos_weight,
    )

    raw_loss = criterion(logits, targets.float())
    masked_loss = raw_loss * mask.float()

    return masked_loss.sum() / mask.float().sum().clamp(min=1.0)


def masked_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Padding target을 제외하고 metric 계산.
    """
    valid = mask.bool()

    if valid.sum().item() == 0:
        return binary_metrics(
            torch.empty(0, device=logits.device),
            torch.empty(0, device=targets.device),
            threshold=threshold,
        )

    return binary_metrics(
        logits[valid],
        targets[valid],
        threshold=threshold,
    )


def count_binary_targets(samples: list) -> dict[str, float]:
    total = 0
    positives = 0

    for sample in samples:
        y = sample.y
        total += int(y.numel())
        positives += int(y.sum().item())

    negatives = total - positives
    positive_rate = positives / max(total, 1)

    return {
        "num_samples": len(samples),
        "num_targets": total,
        "num_positive_targets": positives,
        "num_negative_targets": negatives,
        "positive_rate": positive_rate,
    }