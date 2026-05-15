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

from src.gnn_dataset import estimate_pos_weight, load_frame_samples, move_frame_sample, split_samples
from src.gnn_models import SingleFrameGATClassifier
from src.training_utils import binary_metrics, ensure_reproducible
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-frame expert-informed GAT blind-zone classifier.")
    parser.add_argument("--graphs", default="outputs/graphs", help="Directory containing graph JSON files")
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
    args = parser.parse_args()

    ensure_reproducible(args.seed)
    samples, metadata = load_frame_samples(args.graphs)
    if not samples:
        raise SystemExit(f"No frame samples with blind-zone labels found in {args.graphs}")
    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
    node_dim = samples[0].x.size(1)
    edge_dim = samples[0].edge_attr.size(1) if samples[0].edge_attr.ndim == 2 and samples[0].edge_attr.numel() else 0
    device = torch.device(args.device)

    model = SingleFrameGATClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    pos_weight = estimate_pos_weight(train_samples).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    print(f"samples: train={len(train_samples)}, val={len(val_samples)}, node_dim={node_dim}, edge_dim={edge_dim}")
    print(f"pos_weight={float(pos_weight.item()):.3f}, device={device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_samples, criterion, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_samples, criterion, None, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.3f} "
            f"val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.3f} val_recall={val_metrics['recall']:.3f}"
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "metadata": metadata,
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "best_val_metrics": val_metrics,
                },
                output_path,
            )

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({"best_val_f1": best_val_f1}, indent=2), encoding="utf-8")
    print(f"[OK] saved checkpoint: {output_path}")


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
        return 0.0, binary_metrics(torch.tensor([]), torch.tensor([]))
    return sum(losses) / len(losses), binary_metrics(torch.cat(all_logits), torch.cat(all_targets))


if __name__ == "__main__":
    main()
