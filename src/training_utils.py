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


def ranking_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    if logits.numel() == 0 or targets.numel() == 0:
        return {"auprc": 0.0, "auroc": 0.0, "best_f1": 0.0, "best_threshold": 0.5}

    probs = torch.sigmoid(logits.detach()).flatten().cpu()
    targets = targets.detach().flatten().float().cpu()
    positives = float(targets.sum().item())
    negatives = float(targets.numel() - positives)
    if positives == 0.0 or negatives == 0.0:
        base = binary_metrics(logits, targets)
        return {"auprc": float(positives > 0.0), "auroc": 0.0, "best_f1": base["f1"], "best_threshold": 0.5}

    order = torch.argsort(probs, descending=True)
    sorted_probs = probs[order]
    sorted_targets = targets[order]
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)
    recall = tp / max(positives, 1.0)
    precision = tp / torch.clamp(tp + fp, min=1.0)

    # Average precision: step-wise area under the precision-recall curve.
    prev_recall = torch.cat([torch.tensor([0.0]), recall[:-1]])
    auprc = float(((recall - prev_recall) * precision).sum().item())

    # ROC AUC using rank statistics, with average ranks for tied probabilities.
    sorted_asc = torch.argsort(probs, descending=False)
    ranks = torch.zeros_like(probs)
    idx = 0
    while idx < probs.numel():
        jdx = idx + 1
        while jdx < probs.numel() and probs[sorted_asc[jdx]] == probs[sorted_asc[idx]]:
            jdx += 1
        avg_rank = (idx + 1 + jdx) / 2.0
        ranks[sorted_asc[idx:jdx]] = avg_rank
        idx = jdx
    pos_rank_sum = float(ranks[targets == 1].sum().item())
    auroc = (pos_rank_sum - positives * (positives + 1.0) / 2.0) / max(positives * negatives, 1.0)

    f1_scores = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    best_idx = int(torch.argmax(f1_scores).item())
    return {
        "auprc": auprc,
        "auroc": float(auroc),
        "best_f1": float(f1_scores[best_idx].item()),
        "best_threshold": float(sorted_probs[best_idx].item()),
    }


def classification_report_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    metrics = binary_metrics(logits, targets, threshold=threshold)
    metrics.update(ranking_metrics(logits, targets))
    return metrics
