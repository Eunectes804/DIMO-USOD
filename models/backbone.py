"""TinyNeXt-S backbone adapted to return four dense feature maps.

Blocks follow the official ICCV 2025 implementation. Classification heads
are omitted; stage outputs are returned for the decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 groups: int = 1) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Stem(nn.Sequential):
    """RGB -> 16 -> 32, with output resolution 1/4."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNReLU(in_channels, out_channels // 2, 3, 2),
            ConvBNReLU(out_channels // 2, out_channels, 3, 2),
        )


class MV2Block(nn.Module):
    """TinyNeXt's MobileNetV2-style local block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 expanded_channels: int) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            ConvBNReLU(in_channels, expanded_channels, kernel_size=1),
            ConvBNReLU(expanded_channels, expanded_channels, stride=stride,
                       groups=expanded_channels),
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.layers(x)
        return x + transformed if self.use_residual else transformed


class Embed(nn.Sequential):
    """Official stride-2 transition between adjacent stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            MV2Block(in_channels, out_channels, stride=2,
                     expanded_channels=out_channels)
        )


class LeanSingleHeadAttention(nn.Module):
    """Single-head attention without a separate query projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = channels**-0.5
        # Names preserve the official linear1 / linear2 checkpoint keys.
        self.linear1 = nn.Linear(channels, channels, bias=False)
        self.linear2 = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        attention = (self.linear1(tokens) @ tokens.transpose(-2, -1))
        attention = (attention * self.scale).softmax(dim=-1)
        tokens = attention @ self.linear2(tokens)
        return tokens.transpose(1, 2).reshape(
            batch, channels, height, width
        ).contiguous()


class PointwiseMLP(nn.Sequential):
    def __init__(self, channels: int, ratio: float) -> None:
        hidden_channels = int(ratio * channels)
        super().__init__(
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )


class FormerBlock(nn.Module):
    """Global attention + local DWConv + MLP at the 1/16 stage."""

    def __init__(self, channels: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.BatchNorm2d(channels), LeanSingleHeadAttention(channels)
        )
        self.local = nn.Conv2d(channels, channels, 3, padding=1,
                               groups=channels, bias=False)
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(channels), PointwiseMLP(channels, mlp_ratio)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = x + self.local(x)
        return x + self.mlp(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class SEBlock(nn.Module):
    """SE channel selection + local DWConv + MLP at the 1/32 stage."""

    def __init__(self, channels: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.BatchNorm2d(channels), SqueezeExcitation(channels, reduction=4)
        )
        self.local = nn.Conv2d(channels, channels, 3, padding=1,
                               groups=channels, bias=False)
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(channels), PointwiseMLP(channels, mlp_ratio)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.se(x)
        x = x + self.local(x)
        return x + self.mlp(x)


def make_stage(name: str, channels: int, depth: int,
               ratio: float) -> nn.Sequential:
    blocks: List[nn.Module] = []
    for _ in range(depth):
        if name == "mv2":
            blocks.append(MV2Block(channels, channels, 1,
                                   int(ratio * channels)))
        elif name == "former":
            blocks.append(FormerBlock(channels, ratio))
        elif name == "se":
            blocks.append(SEBlock(channels, ratio))
        else:
            raise ValueError(f"Unknown TinyNeXt block: {name}")
    return nn.Sequential(*blocks)


class TinyNeXtSBackbone(nn.Module):
    """TinyNeXt-S returning E1/E2/E3/E4 at /4, /8, /16 and /32."""

    OUTPUT_CHANNELS = (32, 64, 96, 192)
    CONFIG = (
        ("mv2", 32, 3, 2.0),
        ("mv2", 64, 3, 2.0),
        ("former", 96, 8, 2.0),
        ("se", 192, 3, 2.0),
    )

    def __init__(self, pretrained_path: Optional[str] = None) -> None:
        super().__init__()
        self.embeds = nn.ModuleList([Stem(3, self.CONFIG[0][1])])
        self.stages = nn.ModuleList()
        input_channels = self.CONFIG[0][1]
        for index, (name, channels, depth, ratio) in enumerate(self.CONFIG):
            if index > 0:
                self.embeds.append(Embed(input_channels, channels))
            self.stages.append(make_stage(name, channels, depth, ratio))
            input_channels = channels
        self.norm = nn.BatchNorm2d(input_channels)
        self._initialize_weights()
        self.pretrained_report: Optional[Dict[str, object]] = None
        if pretrained_path is not None:
            self.pretrained_report = self.load_pretrained(pretrained_path)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm,
                                   nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _unwrap(checkpoint: object) -> Mapping[str, torch.Tensor]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("TinyNeXt checkpoint must contain a state dict")
        for wrapper in ("model", "state_dict", "model_state_dict"):
            candidate = checkpoint.get(wrapper)
            if isinstance(candidate, Mapping):
                checkpoint = candidate
                break
        return {str(k): v for k, v in checkpoint.items()
                if torch.is_tensor(v)}

    @staticmethod
    def _normalize_key(key: str) -> str:
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        # Translate the official SE key to the local module name.
        key = key.replace(".se.1.se.", ".se.1.gate.")
        return key

    def load_pretrained(self, path: str) -> Dict[str, object]:
        weight_path = Path(path)
        if not weight_path.is_file():
            raise FileNotFoundError(f"TinyNeXt-S weights not found: {weight_path}")
        state = self._unwrap(torch.load(weight_path, map_location="cpu"))
        current = self.state_dict()
        matched: Dict[str, torch.Tensor] = {}
        ignored: List[str] = []
        unexpected: List[str] = []
        mismatched: List[str] = []
        for original_key, value in state.items():
            key = self._normalize_key(original_key)
            if key.startswith(("class_head.", "dist_head.", "global_pool.")):
                ignored.append(original_key)
            elif key not in current:
                unexpected.append(original_key)
            elif current[key].shape != value.shape:
                mismatched.append(original_key)
            else:
                matched[key] = value
        missing = sorted(set(current) - set(matched))
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "TinyNeXt-S checkpoint does not fully cover the backbone: "
                f"matched={len(matched)}/{len(current)}, missing={missing}, "
                f"unexpected={unexpected}, mismatched={mismatched}"
            )
        self.load_state_dict(matched, strict=True)
        return {
            "path": str(weight_path.resolve()),
            "matched_tensors": len(matched),
            "backbone_tensors": len(current),
            "coverage": 1.0,
            "ignored_classification_keys": sorted(ignored),
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        for index in range(4):
            x = self.embeds[index](x)
            x = self.stages[index](x)
            if index == 3:
                x = self.norm(x)
            features.append(x)
        return features


__all__ = ["TinyNeXtSBackbone"]
