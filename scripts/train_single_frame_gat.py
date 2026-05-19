#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. In Colab, run: !pip install -r requirements.txt") from exc

from src.gnn_dataset import (
    align_frame_samples,
    alignment_report,
    estimate_pos_weight,
    fit_feature_normalizer,
    normalize_frame_samples,
    load_frame_sample_splits,
    load_frame_samples,
    move_frame_sample,
    stabilize_expert_features,
    split_samples,
)
from src.gnn_models import SingleFrameGATClassifier, SingleFrameMRGCNClassifier
from src.training_utils import classification_report_metrics, ensure_reproducible
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-frame expert-informed GAT blind-zone classifier.")
    parser.add_argument("--model", default="eigat", choices=["eigat", "mrgcn"], help="Single-frame graph model family")
    parser.add_argument("--graphs", default="outputs/graphs", help="Directory containing graph JSON files or graph_dataset.pkl")
    parser.add_argument("--output", default="outputs/models/single_frame_gat.pt", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional debug cap for training samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional debug cap for validation samples")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional debug cap for test samples")
    parser.add_argument("--eval-graphs", default=None, help="Optional held-out graph JSON directory to evaluate after training")
    parser.add_argument("--metrics-output", default=None, help="Optional metrics JSON output path")
    parser.add_argument("--no-normalize", action="store_true", help="Disable train-fitted feature normalization")
    parser.add_argument("--disable-edge-attr", action="store_true", help="Ablation: remove expert edge features")
    parser.add_argument("--train-scenes", nargs="*", default=None)
    parser.add_argument("--val-scenes", nargs="*", default=None)
    parser.add_argument("--test-scenes", nargs="*", default=None)
    parser.add_argument("--scene-split", default=None, help="JSON file with train/val/test scene id lists")
    parser.add_argument("--negative-ratio", type=float, default=None, help="Train-time negative samples per positive sample")
    parser.add_argument("--pos-weight", type=float, default=None, help="Override BCE positive class weight")
    parser.add_argument("--selection-metric", default="f1", choices=["f1", "best_f1", "auprc", "auroc"])
    args = parser.parse_args()

    ensure_reproducible(args.seed)
    scene_split = load_scene_split(args.scene_split)
    train_scene_ids = scene_split.get("train", args.train_scenes)
    val_scene_ids = scene_split.get("val", args.val_scenes)
    test_scene_ids = scene_split.get("test", args.test_scenes)
    graph_path = Path(args.graphs)
    if graph_path.is_file() and graph_path.suffix == ".pkl":
        splits, metadata = load_frame_sample_splits(graph_path)
        samples = splits["train"] + splits["val"] + splits["test"]
        train_samples, val_samples = splits["train"], splits["val"]
        test_samples = splits["test"]
    else:
        samples, metadata = load_frame_samples(graph_path)
        if train_scene_ids or val_scene_ids or test_scene_ids:
            train_samples = filter_by_scene(samples, train_scene_ids)
            val_samples = filter_by_scene(samples, val_scene_ids)
            test_samples = filter_by_scene(samples, test_scene_ids)
        else:
            train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
            test_samples = []
    if not samples:
        raise SystemExit(f"No frame samples with blind-zone labels found in {args.graphs}")
    if not train_samples or not val_samples:
        raise SystemExit("Train and validation splits must both contain frame samples.")
    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_samples = val_samples[: args.max_val_samples]
    if args.max_test_samples is not None:
        test_samples = test_samples[: args.max_test_samples]
    if args.disable_edge_attr:
        train_samples = strip_frame_edge_attr(train_samples)
        val_samples = strip_frame_edge_attr(val_samples)
        test_samples = strip_frame_edge_attr(test_samples)
        samples = strip_frame_edge_attr(samples)
        metadata = {**metadata, "edge_feature_names": []}
    else:
        train_samples = stabilize_expert_features(train_samples, metadata)
        val_samples = stabilize_expert_features(val_samples, metadata)
        test_samples = stabilize_expert_features(test_samples, metadata)
        samples = stabilize_expert_features(samples, metadata)

    normalizer = None
    if not args.no_normalize:
        normalizer = fit_feature_normalizer(train_samples, metadata)
        train_samples = normalize_frame_samples(train_samples, normalizer)
        val_samples = normalize_frame_samples(val_samples, normalizer)
        test_samples = normalize_frame_samples(test_samples, normalizer)

    node_dim = samples[0].x.size(1)
    edge_dim = samples[0].edge_attr.size(1) if samples[0].edge_attr.ndim == 2 and samples[0].edge_attr.numel() else 0
    device = torch.device(args.device)

    if args.model == "mrgcn":
        model = SingleFrameMRGCNClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    else:
        model = SingleFrameGATClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    pos_weight = (
        torch.tensor([args.pos_weight], dtype=torch.float32)
        if args.pos_weight is not None
        else estimate_pos_weight(train_samples)
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_score = -1.0
    history = []
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    print(
        f"samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}, "
        f"model={args.model}, node_dim={node_dim}, edge_dim={edge_dim}",
        flush=True,
    )
    print(f"pos_weight={float(pos_weight.item()):.3f}, device={device}", flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_train_samples = sample_training_subset(train_samples, args.negative_ratio, args.seed + epoch)
        train_loss, train_metrics = run_epoch(model, epoch_train_samples, criterion, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_samples, criterion, None, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.3f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.3f} "
            f"val_best_f1={val_metrics['best_f1']:.3f} val_auprc={val_metrics['auprc']:.3f} "
            f"val_auroc={val_metrics['auroc']:.3f} val_recall={val_metrics['recall']:.3f}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )
        val_score = val_metrics[args.selection_metric]
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "metadata": metadata,
                    "normalizer": normalizer.to_dict() if normalizer else None,
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "best_val_metrics": val_metrics,
                    "selection_metric": args.selection_metric,
                    "best_val_score": best_val_score,
                },
                output_path,
            )

    metrics_path = Path(args.metrics_output) if args.metrics_output else output_path.with_suffix(".metrics.json")
    test_metrics = {}
    if test_samples:
        checkpoint = torch.load(output_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_metrics = run_epoch(model, test_samples, criterion, None, device)
        print(
            f"test_loss={test_loss:.4f} test_f1={test_metrics['f1']:.3f} "
            f"test_best_f1={test_metrics['best_f1']:.3f} test_auprc={test_metrics['auprc']:.3f} "
            f"test_auroc={test_metrics['auroc']:.3f} test_recall={test_metrics['recall']:.3f}",
            flush=True,
        )
    eval_metrics = {}
    eval_alignment = {}
    if args.eval_graphs:
        eval_samples, eval_metadata = load_frame_samples(args.eval_graphs)
        eval_alignment = alignment_report(eval_metadata, metadata)
        eval_samples = align_frame_samples(eval_samples, eval_metadata, metadata)
        if args.disable_edge_attr:
            eval_samples = strip_frame_edge_attr(eval_samples)
        else:
            eval_samples = stabilize_expert_features(eval_samples, metadata)
        if normalizer:
            eval_samples = normalize_frame_samples(eval_samples, normalizer)
        eval_loss, eval_metrics = run_epoch(model, eval_samples, criterion, None, device)
        print(
            f"eval_graphs={args.eval_graphs} eval_samples={len(eval_samples)} "
            f"eval_loss={eval_loss:.4f} eval_f1={eval_metrics['f1']:.3f} "
            f"eval_best_f1={eval_metrics['best_f1']:.3f} eval_auprc={eval_metrics['auprc']:.3f} "
            f"eval_auroc={eval_metrics['auroc']:.3f} eval_recall={eval_metrics['recall']:.3f}",
            flush=True,
        )
    metrics_path.write_text(
        json.dumps(
            {
                "selection_metric": args.selection_metric,
                "best_val_score": best_val_score,
                "normalization": None if args.no_normalize else "train_split_standardization",
                "history": history,
                "test_metrics": test_metrics,
                "eval_metrics": eval_metrics,
                "eval_alignment": eval_alignment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] saved checkpoint: {output_path}", flush=True)


def run_epoch(model, samples, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    all_logits = []
    all_targets = []
    with torch.set_grad_enabled(is_train):
        for sample in samples:
            sample = move_frame_sample(sample, device)
            logits = model(sample)
            loss = criterion(logits, sample.y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            all_logits.append(logits.detach().cpu())
            all_targets.append(sample.y.detach().cpu())
    if not losses:
        return 0.0, classification_report_metrics(torch.tensor([]), torch.tensor([]))
    return sum(losses) / len(losses), classification_report_metrics(torch.cat(all_logits), torch.cat(all_targets))


def strip_frame_edge_attr(samples):
    from src.gnn_dataset import FrameGraphSample

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
            edge_types=sample.edge_types,
        )
        for sample in samples
    ]


def filter_by_scene(samples, scene_ids):
    if not scene_ids:
        return []
    scene_id_set = set(scene_ids)
    return [sample for sample in samples if sample.scene_id in scene_id_set]


def load_scene_split(path):
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_training_subset(samples, negative_ratio, seed):
    if negative_ratio is None:
        return samples
    positives = [sample for sample in samples if float(sample.y.sum().item()) > 0.0]
    negatives = [sample for sample in samples if float(sample.y.sum().item()) == 0.0]
    if not positives or not negatives:
        return samples
    rng = random.Random(seed)
    negative_count = min(len(negatives), max(1, int(len(positives) * negative_ratio)))
    subset = positives + rng.sample(negatives, negative_count)
    rng.shuffle(subset)
    return subset


if __name__ == "__main__":
    main()
