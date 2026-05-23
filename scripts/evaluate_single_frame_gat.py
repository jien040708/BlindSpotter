#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. In Colab, run: !pip install -r requirements.txt") from exc

from src.gnn_dataset import (
    FeatureNormalizer,
    align_frame_samples,
    alignment_report,
    load_frame_sample_splits,
    load_frame_samples,
    move_frame_sample,
    normalize_frame_samples,
    stabilize_expert_features,
)
from src.gnn_models import SingleFrameGATClassifier
from src.training_utils import binary_metrics, classification_report_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained single-frame GAT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graphs", required=True, help="graph_dataset.pkl or canonical graph JSON directory")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
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

    graph_path = Path(args.graphs)
    if graph_path.is_file() and graph_path.suffix == ".pkl":
        splits, metadata = load_frame_sample_splits(graph_path)
        samples = splits[args.split] if args.split != "all" else splits["train"] + splits["val"] + splits["test"]
        alignment = alignment_report(metadata, checkpoint["metadata"])
    else:
        samples, metadata = load_frame_samples(graph_path)
        alignment = alignment_report(metadata, checkpoint["metadata"])
        samples = align_frame_samples(samples, metadata, checkpoint["metadata"])

    samples = stabilize_expert_features(samples, checkpoint["metadata"])
    if checkpoint.get("normalizer"):
        normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        samples = normalize_frame_samples(samples, normalizer)

    logits, targets = collect_outputs(model, samples)
    metrics = classification_report_metrics(logits, targets)
    val_threshold = checkpoint.get("best_val_metrics", {}).get("best_threshold", 0.5)
    metrics_at_val_threshold = binary_metrics(logits, targets, threshold=val_threshold)
    result = {
        "checkpoint": args.checkpoint,
        "graphs": args.graphs,
        "split": args.split,
        "n_targets": int(targets.numel()),
        "positives": int(targets.sum().item()),
        "metrics": metrics,
        "metrics_at_checkpoint_val_threshold": metrics_at_val_threshold,
        "checkpoint_val_threshold": val_threshold,
        "alignment": alignment,
    }

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def collect_outputs(model, samples):
    logits = []
    targets = []
    with torch.no_grad():
        for sample in samples:
            sample = move_frame_sample(sample, "cpu")
            logits.append(model(sample).cpu())
            targets.append(sample.y.cpu())
    if not logits:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(logits), torch.cat(targets)


if __name__ == "__main__":
    main()
