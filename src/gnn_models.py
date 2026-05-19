from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .gnn_dataset import FrameGraphSample, TemporalGraphSample


RELATION_TO_ID = {
    "vru_to_vehicle": 0,
    "vehicle_to_vru": 1,
    "vru_to_blind_zone": 2,
    "blind_zone_to_vru": 3,
    "vehicle_to_blind_zone": 4,
    "blind_zone_to_vehicle": 5,
    "spatial_near": 6,
    "temporal_next": 7,
    "unknown": 8,
}


class ExpertGATLayer(nn.Module):
    """Small edge-aware GAT layer with expert edge features as attention bias."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.node_proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False) if edge_dim > 0 else None
        self.skip = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.edge_bias is not None:
            nn.init.xavier_uniform_(self.edge_bias.weight)
        nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        h = self.node_proj(x).view(num_nodes, self.heads, self.out_dim)
        if edge_index.numel() == 0:
            return F.elu((h.reshape(num_nodes, self.heads * self.out_dim) + self.skip(x)))

        src = edge_index[0]
        dst = edge_index[1]
        scores = (h[src] * self.att_src).sum(dim=-1) + (h[dst] * self.att_dst).sum(dim=-1)
        if self.edge_bias is not None and edge_attr.numel() > 0:
            scores = scores + self.edge_bias(edge_attr)
        scores = F.leaky_relu(scores, negative_slope=0.2)

        out = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        for node_idx in torch.unique(dst):
            mask = dst == node_idx
            alpha = torch.softmax(scores[mask], dim=0)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            message = h[src[mask]] * alpha.unsqueeze(-1)
            out[node_idx] = message.sum(dim=0)

        out = out.reshape(num_nodes, self.heads * self.out_dim)
        return F.elu(out + self.skip(x))


class GATEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        modules = []
        in_dim = node_dim
        for _ in range(layers):
            out_per_head = max(1, hidden_dim // heads)
            modules.append(ExpertGATLayer(in_dim, out_per_head, edge_dim, heads=heads, dropout=dropout))
            in_dim = out_per_head * heads
        self.layers = nn.ModuleList(modules)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index, edge_attr)
        return x


class SingleFrameGATClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GATEncoder(node_dim, edge_dim, hidden_dim, heads, layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sample: FrameGraphSample) -> torch.Tensor:
        node_embeddings = self.encoder(sample.x, sample.edge_index, sample.edge_attr)
        target_embeddings = node_embeddings[sample.target_indices]
        return self.classifier(target_embeddings).squeeze(-1)


class RelationalGraphConvLayer(nn.Module):
    """Relation-specific graph convolution for comparing against edge-aware GAT."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, edge_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.num_relations = num_relations
        self.dropout = dropout
        self.self_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.relation_proj = nn.ModuleList(nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations))
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        out = self.self_proj(x)
        if edge_index.numel() == 0:
            return F.elu(self.norm(out))

        src = edge_index[0]
        dst = edge_index[1]
        messages = torch.zeros(num_nodes, out.size(-1), device=x.device, dtype=x.dtype)
        degrees = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        for relation_id in torch.unique(relation_ids):
            rel = int(relation_id.item())
            if rel < 0 or rel >= self.num_relations:
                rel = RELATION_TO_ID["unknown"]
            mask = relation_ids == relation_id
            rel_messages = self.relation_proj[rel](x[src[mask]])
            if self.edge_proj is not None and edge_attr.numel() > 0:
                rel_messages = rel_messages + self.edge_proj(edge_attr[mask])
            messages.index_add_(0, dst[mask], rel_messages)
            degrees.index_add_(0, dst[mask], torch.ones(mask.sum(), 1, device=x.device, dtype=x.dtype))

        out = out + messages / degrees.clamp_min(1.0)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return F.elu(self.norm(out))


class MRGCNEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        num_relations: int = len(RELATION_TO_ID),
    ):
        super().__init__()
        self.num_relations = num_relations
        self.dropout = dropout
        modules = []
        in_dim = node_dim
        for _ in range(layers):
            modules.append(RelationalGraphConvLayer(in_dim, hidden_dim, num_relations, edge_dim, dropout))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(modules)
        self.output_dim = in_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, relation_ids)
        return x


class SingleFrameMRGCNClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MRGCNEncoder(node_dim, edge_dim, hidden_dim, layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sample: FrameGraphSample) -> torch.Tensor:
        relation_ids = relation_tensor(sample.edge_types, sample.edge_index.size(1), sample.edge_index.device)
        node_embeddings = self.encoder(sample.x, sample.edge_index, sample.edge_attr, relation_ids)
        target_embeddings = node_embeddings[sample.target_indices]
        return self.classifier(target_embeddings).squeeze(-1)


def relation_tensor(edge_types: list[str] | None, edge_count: int, device: torch.device) -> torch.Tensor:
    if not edge_types or len(edge_types) != edge_count:
        edge_types = ["unknown"] * edge_count
    ids = [RELATION_TO_ID.get(edge_type, RELATION_TO_ID["unknown"]) for edge_type in edge_types]
    return torch.tensor(ids, dtype=torch.long, device=device)


class TemporalGATClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        temporal_hidden_dim: int = 64,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GATEncoder(node_dim, edge_dim, hidden_dim, heads, layers, dropout)
        self.gru = nn.GRU(self.encoder.output_dim, temporal_hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sample: TemporalGraphSample) -> torch.Tensor:
        device = sample.y.device
        per_frame_embeddings = []
        per_frame_id_to_idx = []
        for frame in sample.frames:
            embeddings = self.encoder(frame.x, frame.edge_index, frame.edge_attr)
            per_frame_embeddings.append(embeddings)
            per_frame_id_to_idx.append({node_id: idx for idx, node_id in enumerate(frame.node_ids)})

        target_sequences = []
        for target_id in sample.target_node_ids:
            history = []
            for embeddings, id_to_idx in zip(per_frame_embeddings, per_frame_id_to_idx):
                if target_id in id_to_idx:
                    history.append(embeddings[id_to_idx[target_id]])
                else:
                    history.append(torch.zeros(self.encoder.output_dim, device=device))
            target_sequences.append(torch.stack(history, dim=0))

        sequence_tensor = torch.stack(target_sequences, dim=0)
        _, hidden = self.gru(sequence_tensor)
        return self.classifier(hidden[-1]).squeeze(-1)
