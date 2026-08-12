"""Mobile-size DBNet, CTC recognizer and table-structure networks."""

from __future__ import annotations

import math
from typing import Any, NamedTuple


# OOD schedules contain up to 40 people.  With day, name/group, summary and
# header cells, a valid image can contain slightly more than 1,500 candidates.
# Keep headroom above that observed maximum so the decoder cannot impose an
# unavoidable false-negative ceiling.
TABLE_CANDIDATE_TOP_K = 2048
MIN_TABLE_CANDIDATE_TOP_K = 1600


class TableCandidates(NamedTuple):
    """Decoded table centers and relation-head grouping assignments.

    The first two tuple positions remain ``scores`` and flattened ``indices``
    for compatibility with the original decoder and the >1,000-cell stress
    test.  Group IDs are arbitrary labels; equality between IDs defines the
    predicted row/column partition.
    """

    scores: Any
    indices: Any
    points: Any
    quads: Any
    row_groups: Any
    column_groups: Any
    row_embeddings: Any
    column_embeddings: Any


def _cluster_relation_embeddings(vectors, distance_threshold: float):
    """Cluster relation embeddings using the training loss margin.

    Relation training pulls equal labels together and pushes different-label
    centroids at least 1.0 apart.  A 0.5 inference threshold is therefore the
    midpoint of that explicit margin, rather than a coordinate heuristic.
    """

    import torch

    distance_threshold = float(distance_threshold)
    if not math.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("relation distance threshold must be finite and positive")
    if vectors.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=vectors.device)
    source_device = vectors.device
    values = vectors.detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("relation embeddings must contain only finite values")
    # At the 1,500-candidate ceiling this is a bounded 2.25M-element matrix.
    # Vectorized pairwise distances avoid 1,500 device synchronizations and
    # make validation practical while connected components keep group labels
    # independent of candidate score ordering.
    adjacency = torch.cdist(values, values) <= distance_threshold
    assignments = torch.full((len(values),), -1, dtype=torch.long)
    group = 0
    while bool((assignments < 0).any()):
        seed = int((assignments < 0).nonzero()[0])
        component = adjacency[seed].clone()
        while True:
            expanded = adjacency[component].any(dim=0)
            if torch.equal(expanded, component):
                break
            component = expanded
        component &= assignments < 0
        assignments[component] = group
        group += 1
    return assignments.to(source_device)


def decode_table_candidates(
    outputs, *, threshold: float = 0.2, top_k: int = TABLE_CANDIDATE_TOP_K,
    relation_distance_threshold: float = 0.5,
    target_size: tuple[int, int] | None = None,
):
    """Decode NMS centers, corner-offset quads and relation groups.

    ``top_k`` is a hard upper bound and is applied after 3x3 local-maximum
    suppression.  When relation heads are present, grouping comes exclusively
    from their learned embeddings.  Coordinate grouping is retained only as a
    compatibility fallback for legacy heatmap-only exports.  ``target_size``
    scales decoded quads into a target coordinate system *after* native-grid
    NMS, avoiding duplicate peaks caused by upsampling a heatmap.
    """

    import torch
    import torch.nn.functional as functional

    if top_k < 1:
        raise ValueError("table candidate top_k must be positive")
    logits = outputs["cell_heatmap"]
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("cell_heatmap must have shape [batch, 1, height, width]")
    source_height, source_width = logits.shape[-2:]
    height, width = source_height, source_width
    target_height, target_width = target_size or (height, width)
    quad_scale_y = target_height / height
    quad_scale_x = target_width / width
    scores = logits.sigmoid()
    local = functional.max_pool2d(scores, 3, stride=1, padding=1)
    local_scores = torch.where(scores >= local, scores, torch.full_like(scores, -1.0))
    flat_scores = local_scores.flatten(1)
    count = min(int(top_k), flat_scores.shape[1])
    values, indices = torch.topk(flat_scores, count, dim=1)

    corner_map = outputs.get("corner_offsets")
    if corner_map is None:
        # Legacy heatmap-only exports retain center decoding, but their
        # degenerate quads deliberately cannot pass polygon-IoU validation.
        corner_map = logits.new_zeros((logits.shape[0], 8, height, width))
    elif corner_map.ndim != 4 or corner_map.shape[1] != 8:
        raise ValueError("corner_offsets must have shape [batch, 8, height, width]")
    else:
        corner_height, corner_width = corner_map.shape[-2:]
        if (corner_height, corner_width) != (height, width):
            corner_map = functional.interpolate(
                corner_map, size=(height, width), mode="bilinear", align_corners=False,
            )
        corner_map = corner_map.clone()
        corner_map[:, 0::2] *= width / corner_width
        corner_map[:, 1::2] *= height / corner_height

    relation_maps = {}
    for name in ("row_embedding", "column_embedding"):
        relation = outputs.get(name)
        if relation is not None and relation.shape[-2:] != (height, width):
            relation = functional.interpolate(relation, size=(height, width), mode="bilinear", align_corners=False)
        relation_maps[name] = relation

    decoded = []
    for batch_index, (batch_values, batch_indices) in enumerate(zip(values, indices)):
        keep = batch_values >= threshold
        batch_values = batch_values[keep]
        batch_indices = batch_indices[keep]
        points = torch.stack((batch_indices // width, batch_indices % width), dim=1)
        sampled_offsets = corner_map[
            batch_index, :, points[:, 0], points[:, 1]
        ].transpose(0, 1).reshape(-1, 4, 2)
        centers_xy = torch.stack((points[:, 1], points[:, 0]), dim=1).to(sampled_offsets.dtype)
        quads = sampled_offsets + centers_xy[:, None, :]
        if (target_height, target_width) != (height, width):
            quads = quads.clone()
            quads[..., 0] *= quad_scale_x
            quads[..., 1] *= quad_scale_y
        embeddings = {}
        groups = {}
        for axis, coordinate_column in (("row", 0), ("column", 1)):
            relation = relation_maps[f"{axis}_embedding"]
            if relation is None:
                vectors = points[:, coordinate_column:coordinate_column + 1].float()
                grouping_threshold = 3.0
            else:
                vectors = relation[batch_index, :, points[:, 0], points[:, 1]].transpose(0, 1)
                grouping_threshold = relation_distance_threshold
            embeddings[axis] = vectors
            groups[axis] = _cluster_relation_embeddings(vectors, grouping_threshold)
        decoded.append(TableCandidates(
            batch_values, batch_indices, points, quads, groups["row"], groups["column"],
            embeddings["row"], embeddings["column"],
        ))
    return decoded


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("model training requires PyTorch") from exc
    return torch, nn


def _components():
    torch, nn = _torch()

    class ConvBNAct(nn.Sequential):
        def __init__(self, input_channels, output_channels, kernel=3, stride=1, groups=1):
            padding = kernel // 2
            super().__init__(
                nn.Conv2d(input_channels, output_channels, kernel, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.Hardswish(inplace=True),
            )

    class InvertedResidual(nn.Module):
        def __init__(self, input_channels, output_channels, stride, expansion):
            super().__init__()
            hidden = input_channels * expansion
            self.use_residual = stride == 1 and input_channels == output_channels
            self.block = nn.Sequential(
                ConvBNAct(input_channels, hidden, 1),
                ConvBNAct(hidden, hidden, 3, stride, groups=hidden),
                nn.Conv2d(hidden, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
            )

        def forward(self, value):
            output = self.block(value)
            return value + output if self.use_residual else output

    class MobileBackbone(nn.Module):
        def __init__(self, width=1.0):
            super().__init__()
            channels = [max(8, int(value * width)) for value in (16, 24, 40, 80, 112)]
            self.stem = ConvBNAct(3, channels[0], 3, 2)
            self.stage4 = nn.Sequential(InvertedResidual(channels[0], channels[1], 2, 4), InvertedResidual(channels[1], channels[1], 1, 3))
            self.stage8 = nn.Sequential(InvertedResidual(channels[1], channels[2], 2, 4), InvertedResidual(channels[2], channels[2], 1, 3))
            self.stage16 = nn.Sequential(InvertedResidual(channels[2], channels[3], 2, 6), InvertedResidual(channels[3], channels[4], 1, 3))
            self.channels = (channels[1], channels[2], channels[4])

        def forward(self, value):
            value = self.stem(value)
            feature4 = self.stage4(value)
            feature8 = self.stage8(feature4)
            feature16 = self.stage16(feature8)
            return feature4, feature8, feature16

    class FPN(nn.Module):
        def __init__(self, channels, output=96):
            super().__init__()
            self.lateral = nn.ModuleList([nn.Conv2d(channel, output, 1) for channel in channels])
            self.smooth4 = ConvBNAct(output, output)
            self.smooth8 = ConvBNAct(output, output)

        def forward(self, features):
            import torch.nn.functional as functional

            p4, p8, p16 = [layer(feature) for layer, feature in zip(self.lateral, features)]
            p8 = self.smooth8(p8 + functional.interpolate(p16, size=p8.shape[-2:], mode="bilinear", align_corners=False))
            p4 = self.smooth4(p4 + functional.interpolate(p8, size=p4.shape[-2:], mode="bilinear", align_corners=False))
            return p4, p8, p16

    return torch, nn, ConvBNAct, MobileBackbone, FPN


class DBNet:
    def __new__(cls, width: float = 0.75):
        torch, nn, ConvBNAct, MobileBackbone, FPN = _components()

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = MobileBackbone(width)
                self.fpn = FPN(self.backbone.channels, 96)
                self.probability = nn.Sequential(ConvBNAct(96, 48), nn.Conv2d(48, 1, 1), nn.Sigmoid())
                self.threshold = nn.Sequential(ConvBNAct(96, 48), nn.Conv2d(48, 1, 1), nn.Sigmoid())

            def forward(self, image):
                feature = self.fpn(self.backbone(image))[0]
                probability = self.probability(feature)
                threshold = self.threshold(feature)
                binary = torch.sigmoid(50.0 * (probability - threshold))
                return {"probability": probability, "threshold": threshold, "binary": binary}

        return Network()


class Recognizer:
    def __new__(cls, class_count: int, width: float = 0.75, hidden_size: int = 192):
        torch, nn, ConvBNAct, MobileBackbone, _FPN = _components()

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = MobileBackbone(width)
                channels = self.backbone.channels[-1]
                self.sequence_projection = nn.Conv2d(channels, hidden_size, 1)
                self.recurrent = nn.LSTM(hidden_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
                self.classifier = nn.Linear(hidden_size * 2, class_count)

            def forward(self, image):
                _feature4, _feature8, feature16 = self.backbone(image)
                feature = self.sequence_projection(feature16).mean(dim=2).transpose(1, 2)
                sequence, _state = self.recurrent(feature)
                return self.classifier(sequence).log_softmax(dim=-1)

        return Network()


class TableStructureModel:
    def __new__(
        cls, width: float = 0.75, attention: bool = False,
        top_k: int = TABLE_CANDIDATE_TOP_K,
    ):
        if top_k < MIN_TABLE_CANDIDATE_TOP_K:
            raise ValueError(
                f"table candidate top_k must be at least {MIN_TABLE_CANDIDATE_TOP_K}"
            )
        torch, nn, ConvBNAct, MobileBackbone, FPN = _components()

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.top_k = top_k
                self.backbone = MobileBackbone(width)
                self.fpn = FPN(self.backbone.channels, 96)
                self.attention = nn.MultiheadAttention(96, 4, batch_first=True) if attention else None
                self.cell_heatmap = nn.Sequential(ConvBNAct(96, 64), nn.Conv2d(64, 1, 1))
                self.corner_offsets = nn.Sequential(ConvBNAct(96, 64), nn.Conv2d(64, 8, 1))
                self.row_embedding = nn.Sequential(ConvBNAct(96, 48), nn.Conv2d(48, 8, 1))
                self.column_embedding = nn.Sequential(ConvBNAct(96, 48), nn.Conv2d(48, 8, 1))

            def forward(self, image):
                feature4, feature8, feature16 = self.backbone(image)
                p4, _p8, p16 = self.fpn((feature4, feature8, feature16))
                if self.attention is not None:
                    batch, channels, height, width_ = p16.shape
                    tokens = p16.flatten(2).transpose(1, 2)
                    tokens, _weights = self.attention(tokens, tokens, tokens, need_weights=False)
                    p16 = tokens.transpose(1, 2).reshape(batch, channels, height, width_)
                    import torch.nn.functional as functional
                    p4 = p4 + functional.interpolate(p16, size=p4.shape[-2:], mode="bilinear", align_corners=False)
                return {
                    "cell_heatmap": self.cell_heatmap(p4),
                    "corner_offsets": self.corner_offsets(p4),
                    "row_embedding": self.row_embedding(p4),
                    "column_embedding": self.column_embedding(p4),
                }

            def decode_candidates(
                self, outputs, threshold: float = 0.2,
                relation_distance_threshold: float = 0.5,
                target_size: tuple[int, int] | None = None,
            ):
                return decode_table_candidates(
                    outputs, threshold=threshold, top_k=self.top_k,
                    relation_distance_threshold=relation_distance_threshold,
                    target_size=target_size,
                )

        return Network()


def build_model(
    kind: str, *, class_count: int = 0, attention: bool = False,
    top_k: int = TABLE_CANDIDATE_TOP_K,
):
    if kind == "dbnet":
        return DBNet()
    if kind == "recognizer":
        if class_count < 2:
            raise ValueError("recognizer class_count must include blank and characters")
        return Recognizer(class_count)
    if kind == "table":
        return TableStructureModel(attention=attention, top_k=top_k)
    raise ValueError(f"unknown model kind: {kind}")
