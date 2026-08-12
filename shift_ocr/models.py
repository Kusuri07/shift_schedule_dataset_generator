"""Mobile-size DBNet, CTC recognizer and table-structure networks."""

from __future__ import annotations

from dataclasses import dataclass


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
    def __new__(cls, width: float = 0.75, attention: bool = False, top_k: int = 1500):
        if top_k < 1200:
            raise ValueError("table candidate top_k must be at least 1200")
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

            def decode_candidates(self, outputs, threshold: float = 0.2):
                scores = outputs["cell_heatmap"].sigmoid().flatten(1)
                count = min(self.top_k, scores.shape[1])
                values, indices = torch.topk(scores, count, dim=1)
                return [(value[mask], index[mask]) for value, index, mask in zip(values, indices, values >= threshold)]

        return Network()


def build_model(kind: str, *, class_count: int = 0, attention: bool = False, top_k: int = 1500):
    if kind == "dbnet":
        return DBNet()
    if kind == "recognizer":
        if class_count < 2:
            raise ValueError("recognizer class_count must include blank and characters")
        return Recognizer(class_count)
    if kind == "table":
        return TableStructureModel(attention=attention, top_k=top_k)
    raise ValueError(f"unknown model kind: {kind}")
