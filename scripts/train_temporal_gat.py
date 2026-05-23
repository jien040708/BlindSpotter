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
    estimate_pos_weight,
    fit_feature_normalizer,
    load_temporal_samples,
    move_temporal_sample,
    normalize_temporal_samples,
    stabilize_temporal_expert_features,
    split_samples,
)
from src.gnn_models import TemporalGATClassifier, TemporalMRGCNTransformerClassifier, TemporalSocialSTGCNClassifier
from src.training_utils import classification_report_metrics, ensure_reproducible
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a temporal expert-informed GAT + GRU blind-zone classifier.")
    parser.add_argument(
        "--temporal-model",
        default="gat_gru",
        choices=["gat_gru", "social_stgcn", "mrgcn_transformer"],
        help="Temporal model family",
    )
    parser.add_argument("--graphs", default="outputs/graphs", help="Directory containing graph JSON files")
    parser.add_argument("--output", default="outputs/models/temporal_gat.pt", help="Checkpoint output path")
    parser.add_argument("--history", type=int, default=5, help="Number of past/current frames per sample")
    parser.add_argument("--prediction-horizon", type=int, default=0, help="Future frame offset for blind_y target")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-scenes", nargs="*", default=None)
    parser.add_argument("--val-scenes", nargs="*", default=None)
    parser.add_argument("--test-scenes", nargs="*", default=None)
    parser.add_argument("--scene-split", default=None, help="JSON file with train/val/test scene id lists")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--disable-edge-attr", action="store_true", help="Ablation: remove expert edge features")
    parser.add_argument("--negative-ratio", type=float, default=None, help="Train-time negative samples per positive sample")
    parser.add_argument("--pos-weight", type=float, default=None, help="Override BCE positive class weight")
    parser.add_argument("--selection-metric", default="f1", choices=["f1", "best_f1", "auprc", "auroc"])
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional debug cap for training samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional debug cap for validation samples")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional debug cap for test samples")
    args = parser.parse_args()

    ensure_reproducible(args.seed)
    scene_split = load_scene_split(args.scene_split)
    train_scene_ids = scene_split.get("train", args.train_scenes)
    val_scene_ids = scene_split.get("val", args.val_scenes)
    test_scene_ids = scene_split.get("test", args.test_scenes)
    samples, metadata = load_temporal_samples(args.graphs, history=args.history, prediction_horizon=args.prediction_horizon)
    if not samples:
        raise SystemExit(f"No temporal samples with blind-zone labels found in {args.graphs}")
    if train_scene_ids or val_scene_ids or test_scene_ids:
        train_samples = filter_by_scene(samples, train_scene_ids)
        val_samples = filter_by_scene(samples, val_scene_ids)
        test_samples = filter_by_scene(samples, test_scene_ids)
    else:
        train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
        test_samples = []
    if args.disable_edge_attr:
        train_samples = strip_temporal_edge_attr(train_samples)
        val_samples = strip_temporal_edge_attr(val_samples)
        test_samples = strip_temporal_edge_attr(test_samples)
        metadata = {**metadata, "edge_feature_names": []}
    else:
        train_samples = stabilize_temporal_expert_features(train_samples, metadata)
        val_samples = stabilize_temporal_expert_features(val_samples, metadata)
        test_samples = stabilize_temporal_expert_features(test_samples, metadata)
    if not train_samples or not val_samples:
        raise SystemExit("Train and validation splits must both contain temporal samples.")
    if args.max_train_samples is not None:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_samples = val_samples[: args.max_val_samples]
    if args.max_test_samples is not None:
        test_samples = test_samples[: args.max_test_samples]
    normalizer = None
    if not args.no_normalize:
        normalizer = fit_feature_normalizer([frame for sample in train_samples for frame in sample.frames], metadata)
        train_samples = normalize_temporal_samples(train_samples, normalizer)
        val_samples = normalize_temporal_samples(val_samples, normalizer)
        test_samples = normalize_temporal_samples(test_samples, normalizer)
    first_frame = samples[0].frames[-1]
    node_dim = first_frame.x.size(1)
    edge_dim = first_frame.edge_attr.size(1) if first_frame.edge_attr.ndim == 2 and first_frame.edge_attr.numel() else 0
    if args.disable_edge_attr:
        edge_dim = 0
    device = torch.device(args.device)

    if args.temporal_model == "social_stgcn":
        model = TemporalSocialSTGCNClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            temporal_hidden_dim=args.temporal_hidden_dim,
            layers=args.layers,
            dropout=args.dropout,
        ).to(device)
    elif args.temporal_model == "mrgcn_transformer":
        model = TemporalMRGCNTransformerClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            temporal_hidden_dim=args.temporal_hidden_dim,
            heads=args.heads,
            layers=args.layers,
            dropout=args.dropout,
            max_history=max(args.history, 64),
        ).to(device)
    else:
        model = TemporalGATClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            temporal_hidden_dim=args.temporal_hidden_dim,
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
        f"history={args.history}, horizon={args.prediction_horizon}, model={args.temporal_model}, "
        f"node_dim={node_dim}, edge_dim={edge_dim}",
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
    metrics_path.write_text(
        json.dumps(
            {
                "selection_metric": args.selection_metric,
                "best_val_score": best_val_score,
                "test_metrics": test_metrics,
                "normalization": None if args.no_normalize else "train_split_standardization",
                "history": history,
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
            sample = move_temporal_sample(sample, device)
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

            # 명시적으로 GPU 메모리 해제
            del sample, logits, loss
            torch.cuda.empty_cache()
    if not losses:
        return 0.0, classification_report_metrics(torch.tensor([]), torch.tensor([]))
    return sum(losses) / len(losses), classification_report_metrics(torch.cat(all_logits), torch.cat(all_targets))


def filter_by_scene(samples, scene_ids):
    if not scene_ids:
        return []
    scene_id_set = set(scene_ids)
    return [sample for sample in samples if sample.scene_id in scene_id_set]


def strip_temporal_edge_attr(samples):
    stripped = []
    from src.gnn_dataset import FrameGraphSample, TemporalGraphSample

    for sample in samples:
        frames = [
            FrameGraphSample(
                scene_id=frame.scene_id,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                node_ids=frame.node_ids,
                node_types=frame.node_types,
                x=frame.x,
                edge_index=frame.edge_index,
                edge_attr=torch.empty(frame.edge_index.size(1), 0),
                target_indices=frame.target_indices,
                y=frame.y,
                edge_types=frame.edge_types,
            )
            for frame in sample.frames
        ]
        stripped.append(
            TemporalGraphSample(
                scene_id=sample.scene_id,
                frame_id=sample.frame_id,
                timestamp=sample.timestamp,
                frames=frames,
                target_node_ids=sample.target_node_ids,
                y=sample.y,
            )
        )
    return stripped


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
