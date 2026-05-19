from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from blindspot_risk.gnn_dataset import (
    collate_temporal_graphs,
    load_temporal_samples_from_graphs,
)
from blindspot_risk.gnn_models import SimpleSTGCNNClassifier
from blindspot_risk.training_utils import masked_binary_metrics
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
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["test"])
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {})

    history = args.history if args.history is not None else int(config.get("history", 5))

    graphs = read_graphs(args.graph_dir, args.splits)
    samples, metadata = load_temporal_samples_from_graphs(graphs, history=history)

    if not samples:
        raise RuntimeError("No evaluation samples were created.")

    node_dim = int(ckpt["node_dim"])

    model = SimpleSTGCNNClassifier(
        node_dim=node_dim,
        hidden_dim=int(config.get("hidden_dim", 64)),
        temporal_hidden_dim=int(config.get("temporal_hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_logits = []
    all_targets = []
    all_masks = []

    csv_path = out_dir / "risk_predictions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "frame_id",
                "timestamp",
                "target_node_id",
                "risk_probability",
                "prediction",
                "label",
            ],
        )
        writer.writeheader()

        for sample in samples:
            batch = collate_temporal_graphs([sample])
            batch = move_batch_to_device(batch, device)

            logits = model(
                x=batch["x"],
                adj=batch["adj"],
                target_indices=batch["target_indices"],
                node_mask=batch["node_mask"],
            )

            probs = torch.sigmoid(logits).detach().cpu()[0]
            labels = batch["y"].detach().cpu()[0]
            mask = batch["target_mask"].detach().cpu()[0]
            logits_cpu = logits.detach().cpu()[0]

            all_logits.append(logits_cpu.unsqueeze(0))
            all_targets.append(labels.unsqueeze(0))
            all_masks.append(mask.unsqueeze(0))

            for idx, target_node_id in enumerate(sample.target_node_ids):
                if mask[idx].item() == 0:
                    continue

                prob = float(probs[idx].item())
                pred = int(prob >= args.threshold)
                label = int(labels[idx].item())

                writer.writerow(
                    {
                        "scene_id": sample.scene_id,
                        "frame_id": sample.frame_id,
                        "timestamp": sample.timestamp,
                        "target_node_id": target_node_id,
                        "risk_probability": prob,
                        "prediction": pred,
                        "label": label,
                    }
                )

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    metrics = masked_binary_metrics(
        logits=logits,
        targets=targets,
        mask=masks,
        threshold=args.threshold,
    )

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    info(f"Saved risk predictions to {csv_path}")
    info(f"Saved metrics to {metrics_path}")
    info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()