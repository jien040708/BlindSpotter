#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. In Colab, run: !pip install -r requirements.txt") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.gnn_dataset import (  # noqa: E402
    FeatureNormalizer,
    FrameGraphSample,
    align_frame_samples,
    load_frame_sample_splits,
    load_frame_samples,
    move_frame_sample,
    normalize_frame_samples,
    stabilize_expert_features,
)
from src.gnn_models import SingleFrameGATClassifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC/PR/F1 diagnostics for trained blind-zone classifiers.")
    parser.add_argument("--output", default="outputs/figures/model_diagnostics.png")
    parser.add_argument("--pdf-output", default="outputs/figures/model_diagnostics.pdf")
    args = parser.parse_args()

    experiments = [
        {
            "name": "Generated canonical EIGAT",
            "checkpoint": "outputs/models/single_frame_gat_canonical_stable_1layer_5ep.pt",
            "graphs": "outputs/models/graph_dataset_canonical.pkl",
            "split": "test",
            "color": "#2563eb",
        },
        {
            "name": "Real IMPTC set01 EIGAT",
            "checkpoint": "outputs/models/single_frame_gat_imptc_set01_balanced_20ep.pt",
            "graphs": "outputs/graphs_imptc_set01",
            "scene_split": "outputs/splits/imptc_set01_scene_split.json",
            "split": "test",
            "color": "#dc2626",
        },
    ]

    results = []
    for experiment in experiments:
        if not Path(experiment["checkpoint"]).exists() or not Path(experiment["graphs"]).exists():
            print(f"[WARN] skipped missing experiment: {experiment['name']}")
            continue
        logits, targets = collect_predictions(experiment)
        results.append({**experiment, **curve_data(logits, targets)})

    if not results:
        raise SystemExit("No available experiments to plot.")

    output = Path(args.output)
    pdf_output = Path(args.pdf_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16.5, 11), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    plot_roc(axes[0], results)
    plot_pr(axes[1], results)
    plot_f1_threshold(axes[2], results)
    plot_score_distribution(axes[3], results)
    fig.suptitle("Blind-Zone Risk Classifier Diagnostics", fontsize=22, weight="bold", color="#0f172a", y=0.985)
    fig.text(
        0.5,
        0.012,
        "ROC/PR curves show ranking quality; F1-threshold and score histograms show why rare-event models need threshold calibration.",
        ha="center",
        fontsize=10.8,
        color="#475569",
    )

    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_output, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[OK] saved {output}")
    print(f"[OK] saved {pdf_output}")


def collect_predictions(experiment: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(experiment["checkpoint"], map_location="cpu")
    model = SingleFrameGATClassifier(
        node_dim=checkpoint["node_dim"],
        edge_dim=checkpoint["edge_dim"],
        hidden_dim=checkpoint["args"]["hidden_dim"],
        heads=checkpoint["args"]["heads"],
        layers=checkpoint["args"]["layers"],
        dropout=checkpoint["args"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    graph_path = Path(experiment["graphs"])
    if graph_path.is_file() and graph_path.suffix == ".pkl":
        splits, metadata = load_frame_sample_splits(graph_path)
        samples = splits[experiment.get("split", "test")]
    else:
        samples, metadata = load_frame_samples(graph_path)
        split_path = experiment.get("scene_split")
        if split_path:
            scene_ids = json.loads(Path(split_path).read_text(encoding="utf-8"))[experiment.get("split", "test")]
            samples = [sample for sample in samples if sample.scene_id in set(scene_ids)]
        samples = align_frame_samples(samples, metadata, checkpoint["metadata"])

    if checkpoint["metadata"].get("edge_feature_names"):
        samples = stabilize_expert_features(samples, checkpoint["metadata"])
    else:
        samples = strip_edge_attr(samples)

    if checkpoint.get("normalizer"):
        samples = normalize_frame_samples(samples, FeatureNormalizer.from_dict(checkpoint["normalizer"]))

    logits = []
    targets = []
    with torch.no_grad():
        for sample in samples:
            moved = move_frame_sample(sample, "cpu")
            logits.append(model(moved).cpu())
            targets.append(sample.y.cpu())
    return torch.cat(logits), torch.cat(targets)


def strip_edge_attr(samples: list[FrameGraphSample]) -> list[FrameGraphSample]:
    return [
        FrameGraphSample(
            scene_id=sample.scene_id,
            frame_id=sample.frame_id,
            timestamp=sample.timestamp,
            node_ids=sample.node_ids,
            node_types=sample.node_types,
            x=sample.x,
            edge_index=sample.edge_index,
            edge_attr=torch.empty(sample.edge_index.size(1), 0),
            target_indices=sample.target_indices,
            y=sample.y,
        )
        for sample in samples
    ]


def curve_data(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    scores = torch.sigmoid(logits).flatten().numpy()
    y = targets.flatten().numpy().astype(np.float64)
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_y = y[order]
    positives = max(sorted_y.sum(), 1.0)
    negatives = max(len(sorted_y) - sorted_y.sum(), 1.0)

    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1.0 - sorted_y)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1.0)
    fpr = fp / negatives
    tpr = recall

    roc_x = np.concatenate([[0.0], fpr, [1.0]])
    roc_y = np.concatenate([[0.0], tpr, [1.0]])
    auroc = float(np.trapz(roc_y, roc_x))

    pr_recall = np.concatenate([[0.0], recall])
    pr_precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    auprc = float(np.sum((recall - np.concatenate([[0.0], recall[:-1]])) * precision))

    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.argmax(f1)) if len(f1) else 0
    positive_rate = float(y.mean()) if len(y) else 0.0

    return {
        "scores": scores,
        "targets": y,
        "roc_x": roc_x,
        "roc_y": roc_y,
        "pr_recall": pr_recall,
        "pr_precision": pr_precision,
        "thresholds": sorted_scores,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": float(f1[best_idx]) if len(f1) else 0.0,
        "best_threshold": float(sorted_scores[best_idx]) if len(sorted_scores) else 0.5,
        "positive_rate": positive_rate,
        "n": int(len(y)),
        "positives": int(y.sum()),
    }


def style_axis(ax, title: str, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=14.5, weight="bold", color="#0f172a")
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    if ylabel:
        ax.set_ylabel(ylabel, color="#334155")
    if xlabel:
        ax.set_xlabel(xlabel, color="#334155")


def plot_roc(ax, results: list[dict[str, Any]]) -> None:
    style_axis(ax, "A. ROC Curve", "True positive rate", "False positive rate")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.4, label="Random")
    for result in results:
        ax.plot(result["roc_x"], result["roc_y"], color=result["color"], linewidth=2.4, label=f"{result['name']} (AUROC={result['auroc']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)


def plot_pr(ax, results: list[dict[str, Any]]) -> None:
    style_axis(ax, "B. Precision-Recall Curve", "Precision", "Recall")
    for result in results:
        ax.hlines(result["positive_rate"], 0, 1, color=result["color"], linestyle=":", linewidth=1.2, alpha=0.6)
        ax.plot(
            result["pr_recall"],
            result["pr_precision"],
            color=result["color"],
            linewidth=2.4,
            label=f"{result['name']} (AUPRC={result['auprc']:.3f}, base={result['positive_rate']:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)


def plot_f1_threshold(ax, results: list[dict[str, Any]]) -> None:
    style_axis(ax, "C. F1 vs Decision Threshold", "F1-score", "Predicted risk threshold")
    for result in results:
        thresholds = result["thresholds"]
        f1 = result["f1"]
        ax.plot(thresholds, f1, color=result["color"], linewidth=2.2, label=f"{result['name']} best={result['best_f1']:.3f} @ {result['best_threshold']:.3f}")
        ax.scatter([result["best_threshold"]], [result["best_f1"]], color=result["color"], s=45, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.35, max(result["best_f1"] for result in results) + 0.08))
    ax.legend(frameon=False, fontsize=9)


def plot_score_distribution(ax, results: list[dict[str, Any]]) -> None:
    style_axis(ax, "D. Predicted Score Distribution", "Density", "Predicted risk score")
    bins = np.linspace(0, 1, 32)
    for result in results:
        scores = result["scores"]
        targets = result["targets"]
        positive_scores = scores[targets == 1]
        negative_scores = scores[targets == 0]
        ax.hist(negative_scores, bins=bins, density=True, histtype="step", linewidth=1.8, color=result["color"], alpha=0.45, linestyle="--")
        ax.hist(positive_scores, bins=bins, density=True, histtype="step", linewidth=2.2, color=result["color"], label=f"{result['name']} positives")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=9)


if __name__ == "__main__":
    main()
