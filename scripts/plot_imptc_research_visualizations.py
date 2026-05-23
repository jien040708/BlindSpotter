#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gnn_dataset import (
    FeatureNormalizer,
    fit_feature_normalizer,
    load_frame_samples,
    load_temporal_samples,
    move_frame_sample,
    move_temporal_sample,
    normalize_frame_samples,
    normalize_temporal_samples,
    stabilize_expert_features,
    stabilize_temporal_expert_features,
)
from src.gnn_models import (
    SingleFrameGATClassifier,
    SingleFrameMRGCNClassifier,
    TemporalGATClassifier,
    TemporalMRGCNTransformerClassifier,
    TemporalSocialSTGCNClassifier,
)
from src.training_utils import classification_report_metrics


RUN_COLORS = {
    "EIGAT": "#2563eb",
    "MR-GCN": "#16a34a",
    "ST-GCN": "#dc2626",
}


@dataclass
class PredictionTable:
    label: str
    probs: torch.Tensor
    targets: torch.Tensor
    scene_ids: list[str]
    frame_ids: list[str]
    timestamps: list[float]
    target_ids: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style IMPTC diagnostics from trained checkpoints.")
    parser.add_argument("--graphs", default="outputs/graphs_imptc_set01_set02")
    parser.add_argument("--scene-split", default="outputs/splits/imptc_set01_set02_scene_split.json")
    parser.add_argument("--eigat", default="outputs/models/imptc_set01_set02/eigat_single_frame.pt")
    parser.add_argument("--mrgcn", default="outputs/models/imptc_set01_set02/mrgcn_single_frame.pt")
    parser.add_argument("--stgcn", default="outputs/models/imptc_set01_set02/stgcn_temporal_h5_t1.pt")
    parser.add_argument("--output-dir", default="outputs/figures/imptc_set01_set02/research")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_split = json.loads(Path(args.scene_split).read_text(encoding="utf-8"))
    split_scenes = set(scene_split[args.split])

    predictions: list[PredictionTable] = []
    for label, checkpoint_path in [("EIGAT", args.eigat), ("MR-GCN", args.mrgcn)]:
        path = Path(checkpoint_path)
        if path.exists():
            predictions.append(predict_single_frame(label, path, Path(args.graphs), split_scenes))
    stgcn_path = Path(args.stgcn)
    stgcn_predictions = None
    if stgcn_path.exists():
        stgcn_predictions = predict_temporal("ST-GCN", stgcn_path, Path(args.graphs), split_scenes)
        predictions.append(stgcn_predictions)

    if not predictions:
        raise SystemExit("No checkpoints found. Train at least one model first.")

    plot_pr_roc(predictions, output_dir / "pr_roc_curves.png")
    plot_threshold_sweep(predictions, output_dir / "threshold_sweep.png")
    plot_score_histogram(predictions, output_dir / "prediction_score_histogram.png")
    plot_scene_distribution(Path(args.graphs) / "preprocess_summary.json", output_dir / "scene_positive_distribution.png")
    plot_graph_sample(Path(args.graphs), output_dir / "representative_graph_sample.png")
    if stgcn_predictions is not None:
        plot_temporal_timeline(stgcn_predictions, output_dir / "temporal_event_timeline.png")

    print(f"[OK] wrote research visualizations to {output_dir}")


def predict_single_frame(label: str, checkpoint_path: Path, graph_dir: Path, split_scenes: set[str]) -> PredictionTable:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})
    samples, metadata = load_frame_samples(graph_dir)
    samples = [sample for sample in samples if sample.scene_id in split_scenes]
    samples = stabilize_expert_features(samples, checkpoint["metadata"])
    if checkpoint.get("normalizer"):
        samples = normalize_frame_samples(samples, FeatureNormalizer.from_dict(checkpoint["normalizer"]))

    if args.get("model") == "mrgcn" or label == "MR-GCN":
        has_degree_embedding = any("degree_embedding" in key for key in checkpoint["model_state_dict"])
        model = SingleFrameMRGCNClassifier(
            node_dim=checkpoint["node_dim"],
            edge_dim=checkpoint["edge_dim"],
            hidden_dim=args.get("hidden_dim", 32),
            layers=args.get("layers", 1),
            dropout=args.get("dropout", 0.1),
            degree_embedding=has_degree_embedding,
        )
    else:
        model = SingleFrameGATClassifier(
            node_dim=checkpoint["node_dim"],
            edge_dim=checkpoint["edge_dim"],
            hidden_dim=args.get("hidden_dim", 32),
            heads=args.get("heads", 2),
            layers=args.get("layers", 1),
            dropout=args.get("dropout", 0.1),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = []
    targets = []
    scene_ids = []
    frame_ids = []
    timestamps = []
    target_ids = []
    with torch.no_grad():
        for sample in samples:
            sample = move_frame_sample(sample, "cpu")
            logits = model(sample)
            sample_probs = torch.sigmoid(logits).flatten().cpu()
            sample_targets = sample.y.flatten().cpu()
            probs.append(sample_probs)
            targets.append(sample_targets)
            for target_index in sample.target_indices.tolist():
                scene_ids.append(sample.scene_id)
                frame_ids.append(sample.frame_id)
                timestamps.append(sample.timestamp)
                target_ids.append(sample.node_ids[int(target_index)])
    return PredictionTable(label, torch.cat(probs), torch.cat(targets), scene_ids, frame_ids, timestamps, target_ids)


def predict_temporal(label: str, checkpoint_path: Path, graph_dir: Path, split_scenes: set[str]) -> PredictionTable:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})
    samples, metadata = load_temporal_samples(
        graph_dir,
        history=args.get("history", 5),
        prediction_horizon=args.get("prediction_horizon", 1),
    )
    samples = [sample for sample in samples if sample.scene_id in split_scenes]
    samples = stabilize_temporal_expert_features(samples, checkpoint["metadata"])
    if checkpoint.get("normalizer"):
        samples = normalize_temporal_samples(samples, FeatureNormalizer.from_dict(checkpoint["normalizer"]))

    temporal_model = args.get("temporal_model", "gat_gru")
    if temporal_model == "social_stgcn":
        model = TemporalSocialSTGCNClassifier(
            node_dim=checkpoint["node_dim"],
            edge_dim=checkpoint["edge_dim"],
            hidden_dim=args.get("hidden_dim", 32),
            temporal_hidden_dim=args.get("temporal_hidden_dim", 32),
            layers=args.get("layers", 1),
            dropout=args.get("dropout", 0.1),
        )
    elif temporal_model == "mrgcn_transformer":
        model = TemporalMRGCNTransformerClassifier(
            node_dim=checkpoint["node_dim"],
            edge_dim=checkpoint["edge_dim"],
            hidden_dim=args.get("hidden_dim", 32),
            temporal_hidden_dim=args.get("temporal_hidden_dim", 32),
            heads=args.get("heads", 2),
            layers=args.get("layers", 1),
            dropout=args.get("dropout", 0.1),
            max_history=max(args.get("history", 5), 64),
        )
    else:
        model = TemporalGATClassifier(
            node_dim=checkpoint["node_dim"],
            edge_dim=checkpoint["edge_dim"],
            hidden_dim=args.get("hidden_dim", 32),
            temporal_hidden_dim=args.get("temporal_hidden_dim", 32),
            heads=args.get("heads", 2),
            layers=args.get("layers", 1),
            dropout=args.get("dropout", 0.1),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = []
    targets = []
    scene_ids = []
    frame_ids = []
    timestamps = []
    target_ids = []
    with torch.no_grad():
        for sample in samples:
            sample = move_temporal_sample(sample, "cpu")
            logits = model(sample)
            sample_probs = torch.sigmoid(logits).flatten().cpu()
            sample_targets = sample.y.flatten().cpu()
            probs.append(sample_probs)
            targets.append(sample_targets)
            for target_id in sample.target_node_ids:
                scene_ids.append(sample.scene_id)
                frame_ids.append(sample.frame_id)
                timestamps.append(sample.timestamp)
                target_ids.append(target_id)
    return PredictionTable(label, torch.cat(probs), torch.cat(targets), scene_ids, frame_ids, timestamps, target_ids)


def pr_curve(probs: torch.Tensor, targets: torch.Tensor) -> tuple[list[float], list[float], float]:
    order = torch.argsort(probs, descending=True)
    sorted_targets = targets[order].float()
    positives = max(float(sorted_targets.sum().item()), 1.0)
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)
    recall = (tp / positives).tolist()
    precision = (tp / torch.clamp(tp + fp, min=1.0)).tolist()
    metrics = classification_report_metrics(torch.logit(probs.clamp(1e-6, 1 - 1e-6)), targets)
    return [0.0, *recall], [1.0, *precision], metrics["auprc"]


def roc_curve(probs: torch.Tensor, targets: torch.Tensor) -> tuple[list[float], list[float], float]:
    order = torch.argsort(probs, descending=True)
    sorted_targets = targets[order].float()
    positives = max(float(sorted_targets.sum().item()), 1.0)
    negatives = max(float(sorted_targets.numel() - sorted_targets.sum().item()), 1.0)
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)
    tpr = (tp / positives).tolist()
    fpr = (fp / negatives).tolist()
    metrics = classification_report_metrics(torch.logit(probs.clamp(1e-6, 1 - 1e-6)), targets)
    return [0.0, *fpr, 1.0], [0.0, *tpr, 1.0], metrics["auroc"]


def plot_pr_roc(predictions: list[PredictionTable], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for table in predictions:
        color = RUN_COLORS.get(table.label, "#64748b")
        recall, precision, auprc = pr_curve(table.probs, table.targets)
        fpr, tpr, auroc = roc_curve(table.probs, table.targets)
        axes[0].plot(recall, precision, color=color, linewidth=2, label=f"{table.label} AUPRC={auprc:.3f}")
        axes[1].plot(fpr, tpr, color=color, linewidth=2, label=f"{table.label} AUROC={auroc:.3f}")
    base_rate = float(predictions[0].targets.float().mean().item())
    axes[0].axhline(base_rate, color="#94a3b8", linestyle="--", linewidth=1, label=f"base rate={base_rate:.3f}")
    axes[1].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1)
    axes[0].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {output}")


def plot_threshold_sweep(predictions: list[PredictionTable], output: Path) -> None:
    thresholds = torch.linspace(0.0, 1.0, 101)
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.6), sharey=True)
    if len(predictions) == 1:
        axes = [axes]
    for ax, table in zip(axes, predictions):
        precision, recall, f1 = [], [], []
        for threshold in thresholds:
            preds = (table.probs >= threshold).float()
            targets = table.targets.float()
            tp = float(((preds == 1) & (targets == 1)).sum())
            fp = float(((preds == 1) & (targets == 0)).sum())
            fn = float(((preds == 0) & (targets == 1)).sum())
            p = tp / max(tp + fp, 1.0)
            r = tp / max(tp + fn, 1.0)
            precision.append(p)
            recall.append(r)
            f1.append(2 * p * r / max(p + r, 1e-12))
        best_idx = int(torch.tensor(f1).argmax().item())
        ax.plot(thresholds, precision, label="Precision", color="#2563eb")
        ax.plot(thresholds, recall, label="Recall", color="#16a34a")
        ax.plot(thresholds, f1, label="F1", color="#dc2626")
        ax.axvline(float(thresholds[best_idx]), color="#0f172a", linestyle="--", linewidth=1)
        ax.set_title(f"{table.label}: best F1={f1[best_idx]:.3f} @ {thresholds[best_idx]:.2f}")
        ax.set_xlabel("Threshold")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {output}")


def plot_score_histogram(predictions: list[PredictionTable], output: Path) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.6), sharey=True)
    if len(predictions) == 1:
        axes = [axes]
    for ax, table in zip(axes, predictions):
        pos = table.probs[table.targets == 1].numpy()
        neg = table.probs[table.targets == 0].numpy()
        ax.hist(neg, bins=35, range=(0, 1), alpha=0.75, color="#64748b", label=f"negative n={len(neg)}")
        ax.hist(pos, bins=35, range=(0, 1), alpha=0.75, color="#dc2626", label=f"positive n={len(pos)}")
        ax.set_title(f"{table.label} Score Distribution")
        ax.set_xlabel("Predicted risk probability")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {output}")


def plot_scene_distribution(summary_path: Path, output: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scene_ids = [row["scene_id"] for row in summary]
    positives = [int(row["positive_blind_labels"]) for row in summary]
    sets = ["set1" if int(scene_id.split("_")[0]) < 50 else "set2" for scene_id in scene_ids]
    colors = ["#2563eb" if set_name == "set1" else "#dc2626" for set_name in sets]
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.bar(range(len(scene_ids)), positives, color=colors, width=0.85)
    ax.set_title("Scene-Level Positive Blind-Zone Label Distribution")
    ax.set_xlabel("Scene index")
    ax.set_ylabel("Positive blind_y count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#2563eb", lw=6, label="set1"),
            plt.Line2D([0], [0], color="#dc2626", lw=6, label="set2"),
        ],
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {output}")


def plot_graph_sample(graph_dir: Path, output: Path) -> None:
    graph, frame = select_positive_frame(graph_dir)
    x = frame["x"]
    edge_index = frame["edge_index"]
    edge_type = frame.get("edge_type", ["spatial_near"] * len(edge_index[0]))
    target_labels = {int(idx): int(y) for idx, y in zip(frame.get("blind_node_indices", []), frame.get("blind_y", []))}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(f"Representative Frame Graph: {graph['scene_id']} / frame {frame.get('frame_id')}")
    for src, dst, rel in zip(edge_index[0], edge_index[1], edge_type):
        color = "#cbd5e1" if rel == "spatial_near" else "#f97316"
        alpha = 0.22 if rel == "spatial_near" else 0.55
        ax.plot([x[src][0], x[dst][0]], [x[src][1], x[dst][1]], color=color, alpha=alpha, linewidth=0.8)
    for idx, (features, node_type) in enumerate(zip(x, frame["node_types"])):
        if node_type == "occlusion_zone":
            color, marker, size = "#f97316", "D", 100
        elif node_type in {"pedestrian", "cyclist", "e_scooter"}:
            color, marker, size = "#16a34a", "^", 95
        elif node_type == "ego_vehicle":
            color, marker, size = "#2563eb", "s", 140
        else:
            color, marker, size = "#64748b", "o", 70
        edge_color = "#991b1b" if target_labels.get(idx, 0) == 1 else "#ffffff"
        linewidth = 2.7 if idx in target_labels else 0.8
        ax.scatter(features[0], features[1], s=size, c=color, marker=marker, edgecolors=edge_color, linewidths=linewidth, zorder=3)
        if idx in target_labels:
            ax.text(features[0] + 0.4, features[1] + 0.4, f"blind_y={target_labels[idx]}", fontsize=8, color="#991b1b")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {output}")


def select_positive_frame(graph_dir: Path) -> tuple[dict, dict]:
    for path in sorted(graph_dir.glob("*.json")):
        if path.name == "preprocess_summary.json":
            continue
        graph = json.loads(path.read_text(encoding="utf-8"))
        for frame in graph.get("frames", []):
            if sum(int(value) for value in frame.get("blind_y", [])) > 0:
                return graph, frame
    raise SystemExit(f"No positive frame found in {graph_dir}")


def plot_temporal_timeline(table: PredictionTable, output: Path) -> None:
    positive_scenes = []
    for scene_id in sorted(set(table.scene_ids)):
        idxs = [idx for idx, sid in enumerate(table.scene_ids) if sid == scene_id]
        if any(int(table.targets[idx].item()) == 1 for idx in idxs):
            positive_scenes.append((scene_id, idxs))
    if not positive_scenes:
        return
    scene_id, idxs = max(positive_scenes, key=lambda item: sum(int(table.targets[idx].item()) for idx in item[1]))
    idxs = sorted(idxs, key=lambda idx: table.timestamps[idx])
    times = [table.timestamps[idx] for idx in idxs]
    start = min(times)
    times = [time - start for time in times]
    probs = [float(table.probs[idx].item()) for idx in idxs]
    labels = [float(table.targets[idx].item()) for idx in idxs]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(times, probs, color="#dc2626", linewidth=1.8, label="ST-GCN probability")
    ax.scatter(times, labels, color="#0f172a", s=18, alpha=0.65, label="blind_y label")
    ax.set_title(f"Temporal Event Timeline: {scene_id}")
    ax.set_xlabel("Time from first test target [s]")
    ax.set_ylabel("Probability / label")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {output}")


if __name__ == "__main__":
    main()
