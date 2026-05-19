from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from blindspot_risk.gnn_dataset import (
    TemporalGraphDataset,
    collate_temporal_graphs,
    load_temporal_samples_from_graphs,
)
from blindspot_risk.gnn_models import SimpleSTGCNNClassifier
from blindspot_risk.training_utils import (
    count_binary_targets,
    ensure_reproducible,
    masked_bce_with_logits_loss,
    masked_binary_metrics,
)
from blindspot_risk.utils import ensure_dir, info


def read_graphs(graph_dir: str | Path, splits: list[str]) -> list[dict]:
    graph_dir = Path(graph_dir)
    graphs = []

    for split in splits:
        path = graph_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        graphs.append(json.loads(path.read_text(encoding="utf-8")))

    return graphs


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: SimpleSTGCNNClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()

    all_logits = []
    all_targets = []
    all_masks = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(
            x=batch["x"],
            adj=batch["adj"],
            target_indices=batch["target_indices"],
            node_mask=batch["node_mask"],
        )

        all_logits.append(logits.detach().cpu())
        all_targets.append(batch["y"].detach().cpu())
        all_masks.append(batch["target_mask"].detach().cpu())

    if not all_logits:
        return {}

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    return masked_binary_metrics(logits, targets, masks, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--train-splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--val-splits", type=str, nargs="+", default=["eval"])

    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    ensure_reproducible(args.seed)

    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)

    train_graphs = read_graphs(args.graph_dir, args.train_splits)
    val_graphs = read_graphs(args.graph_dir, args.val_splits)

    train_samples, metadata = load_temporal_samples_from_graphs(train_graphs, history=args.history)
    val_samples, _ = load_temporal_samples_from_graphs(val_graphs, history=args.history)

    if not train_samples:
        raise RuntimeError("No training samples were created. Check blind-zone nodes and labels.")
    if not val_samples:
        raise RuntimeError("No validation samples were created. Check eval split.")

    train_summary = count_binary_targets(train_samples)
    val_summary = count_binary_targets(val_samples)

    info("Train target summary:")
    info(json.dumps(train_summary, indent=2))
    info("Validation target summary:")
    info(json.dumps(val_summary, indent=2))

    node_dim = train_samples[0].frames[-1].x.size(1)

    train_loader = DataLoader(
        TemporalGraphDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_temporal_graphs,
    )

    val_loader = DataLoader(
        TemporalGraphDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_temporal_graphs,
    )

    model = SimpleSTGCNNClassifier(
        node_dim=node_dim,
        hidden_dim=args.hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    positives = max(train_summary["num_positive_targets"], 1)
    negatives = max(train_summary["num_negative_targets"], 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = -1.0
    best_epoch = -1

    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            logits = model(
                x=batch["x"],
                adj=batch["adj"],
                target_indices=batch["target_indices"],
                node_mask=batch["node_mask"],
            )

            loss = masked_bce_with_logits_loss(
                logits=logits,
                targets=batch["y"],
                mask=batch["target_mask"],
                pos_weight=pos_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        train_loss = total_loss / max(total_batches, 1)
        val_metrics = evaluate(model, val_loader, device, threshold=args.threshold)

        # Safety-oriented task이므로 AUPRC 우선, 없으면 F1 사용
        score = val_metrics.get("auprc", 0.0)
        if score == 0.0:
            score = val_metrics.get("f1", 0.0)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history_rows.append(row)

        info(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"val_f1={val_metrics.get('f1', 0.0):.4f} | "
            f"val_recall={val_metrics.get('recall', 0.0):.4f} | "
            f"val_auroc={val_metrics.get('auroc', 0.0):.4f} | "
            f"val_auprc={val_metrics.get('auprc', 0.0):.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch

            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "node_dim": node_dim,
                "metadata": metadata,
                "train_summary": train_summary,
                "val_summary": val_summary,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "val_metrics": val_metrics,
            }

            torch.save(ckpt, out_dir / "best_stgcnn.pt")
            info(f"Saved best checkpoint at epoch {epoch}")

    (out_dir / "train_history.json").write_text(
        json.dumps(history_rows, indent=2),
        encoding="utf-8",
    )

    info(f"Training finished. Best epoch={best_epoch}, best_score={best_score:.4f}")


if __name__ == "__main__":
    main()