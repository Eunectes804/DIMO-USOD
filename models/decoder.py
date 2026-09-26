"""DIMO-USOD decoder blocks; GAPNet adaptations are credited in THIRD_PARTY_NOTICES.md."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def feature_map_to_tokens(feature: torch.Tensor) -> torch.Tensor:
    """Convert a feature map to spatial tokens."""

    return feature.flatten(2).transpose(1, 2)


def tokens_to_feature_map(
    tokens: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Restore spatial tokens to a feature map."""

    batch, token_count, channels = tokens.shape
    if token_count != height * width:
        raise ValueError(
            f"Token 数量 {token_count} 与空间尺寸 {height}×{width} 不匹配。"
        )
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def resize_to(
    feature: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """Resize with the decoder's bilinear interpolation convention."""

    # Bilinear upsampling needs FP32 on the supported Windows BF16 runtime.
    interpolation_input = (
        feature.float() if feature.dtype == torch.bfloat16 else feature
    )
    resized = F.interpolate(
        interpolation_input,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.to(dtype=feature.dtype)


class ConvBNReLU(nn.Module):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention shared by GPC and global blocks."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"{channels} 通道不能被 {num_heads} 个注意力头整除。"
            )

        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels**-0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.projection = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, token_count, channels = x.shape

        qkv = self.qkv(x).reshape(
            batch,
            token_count,
            self.num_heads,
            3,
            self.head_channels,
        )
        query, key, value = qkv.permute(3, 0, 2, 1, 4).unbind(0)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        x = attention @ value
        x = x.transpose(1, 2).reshape(batch, token_count, channels)
        return self.projection(x)


class LowResolutionSelfAttention(nn.Module):
    """Apply self-attention after fixed-size spatial pooling."""

    def __init__(
        self,
        channels: int,
        pooled_size: int = 7,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.norm = nn.LayerNorm(channels)
        self.attention = MultiHeadSelfAttention(channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        pooled = self.pool(x)
        height, width = pooled.shape[-2:]
        tokens = feature_map_to_tokens(pooled)
        tokens = self.attention(self.norm(tokens))

        attended = tokens_to_feature_map(tokens, height, width)
        return resize_to(attended, original_size)


class GranularPyramidConvolution(nn.Module):
    """GAPNet GPC with grouped dilations and optional residual attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 40,
        dilations: Sequence[int] = (1, 2, 4, 6),
    ) -> None:
        super().__init__()
        if len(dilations) != 4:
            raise ValueError("GPC 必须为四个通道组提供四个膨胀率。")

        expanded_channels = out_channels * 3 // 2

        first = expanded_channels // 8
        second = expanded_channels // 8
        third = expanded_channels // 4
        fourth = expanded_channels - first - second - third
        self.split_widths = (first, second, third, fourth)

        self.conv1 = nn.Conv2d(
            in_channels,
            expanded_channels,
            1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for group_channels, dilation in zip(self.split_widths, dilations):
            self.convs.append(
                nn.Conv2d(
                    group_channels,
                    group_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    groups=group_channels,
                    bias=False,
                )
            )
            self.bns.append(nn.BatchNorm2d(group_channels))

        self.conv3 = nn.Conv2d(
            expanded_channels,
            out_channels,
            1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.has_residual = in_channels == out_channels
        self.low_resolution_attention: nn.Module = (
            LowResolutionSelfAttention(in_channels)
            if self.has_residual
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.activation(self.bn1(self.conv1(x)))

        split_features = torch.split(expanded, self.split_widths, dim=1)
        processed_groups = [
            self.activation(batch_norm(conv(group)))
            for group, conv, batch_norm in zip(
                split_features,
                self.convs,
                self.bns,
            )
        ]

        output = torch.cat(processed_groups, dim=1)
        output = self.bn3(self.conv3(output))

        # These residuals require matching input and output channels.
        if self.has_residual:
            output = output + x
            output = output + self.low_resolution_attention(x)

        return self.activation(output)


def make_gpc_stack(
    in_channels: int,
    out_channels: int,
    dilations: Sequence[int],
) -> nn.Sequential:
    """Stack a fusion GPC and a refinement GPC."""

    return nn.Sequential(
        GranularPyramidConvolution(
            in_channels,
            out_channels,
            dilations,
        ),
        GranularPyramidConvolution(
            out_channels,
            out_channels,
            dilations,
        ),
    )


class InvertedResidualFFN(nn.Module):
    """Apply pointwise-depthwise-pointwise convolutions to spatial tokens."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_channels, hidden_channels, 1)
        self.activation = nn.Hardswish()
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
        )
        self.project = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        feature = tokens_to_feature_map(tokens, height, width)
        feature = self.activation(self.expand(feature))
        feature = self.activation(self.depthwise(feature))
        feature = self.project(feature)
        return feature_map_to_tokens(feature)


class GlobalTransformerBlock(nn.Module):
    """Self-attention and inverted-residual FFN for global features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels

        self.norm1 = nn.LayerNorm(in_channels)
        self.attention = MultiHeadSelfAttention(in_channels, num_heads)

        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = InvertedResidualFFN(
            in_channels,
            out_channels,
            hidden_channels=int(in_channels * mlp_ratio),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        transformed = self.ffn(self.norm2(tokens), height, width)

        # The FFN residual applies only when its channel count matches.
        return tokens + transformed if self.same_channels else transformed


class MultiHeadCrossAttention(nn.Module):
    """Use lower-stage queries and higher-stage keys and values."""

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"{channels} 通道不能被 {num_heads} 个注意力头整除。"
            )

        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels**-0.5

        self.query = nn.Linear(channels, channels, bias=False)
        self.key_value = nn.Linear(channels, channels * 2, bias=False)
        self.projection = nn.Linear(channels, channels)

    def forward(
        self,
        low_tokens: torch.Tensor,
        high_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, low_count, channels = low_tokens.shape
        high_count = high_tokens.shape[1]

        query = self.query(low_tokens).reshape(
            batch,
            low_count,
            self.num_heads,
            self.head_channels,
        )
        query = query.permute(0, 2, 1, 3)

        key_value = self.key_value(high_tokens).reshape(
            batch,
            high_count,
            self.num_heads,
            2,
            self.head_channels,
        )
        key, value = key_value.permute(3, 0, 2, 1, 4).unbind(0)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        output = attention @ value
        output = output.transpose(1, 2).reshape(
            batch,
            low_count,
            channels,
        )
        return self.projection(output)


class FeedForwardNetwork(nn.Module):
    """Refine cross-scale tokens with a local depthwise convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.activation = nn.Hardswish()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.fc2 = nn.Linear(channels, channels)

    def forward(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        tokens = self.activation(self.fc1(tokens))

        feature = tokens_to_feature_map(tokens, height, width)
        feature = feature + self.depthwise(feature)

        tokens = feature_map_to_tokens(feature)
        return self.fc2(tokens)


class CrossScaleAttention(nn.Module):
    """Update lower-stage features with higher-stage cross-attention."""

    def __init__(
        self,
        channels: int = 40,
        num_heads: int = 8,
        query_residual: bool = False,
    ) -> None:
        super().__init__()
        # Preserve the lower-stage query only when explicitly requested.
        self.query_residual = bool(query_residual)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = MultiHeadCrossAttention(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FeedForwardNetwork(channels)

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
    ) -> torch.Tensor:
        height, width = low_feature.shape[-2:]

        low_tokens = feature_map_to_tokens(low_feature)
        high_tokens = feature_map_to_tokens(high_feature)

        attention_tokens = self.attention(
            self.norm1(low_tokens),
            self.norm1(high_tokens),
        )

        tokens = (
            low_tokens + attention_tokens
            if self.query_residual
            else attention_tokens
        )
        tokens = tokens + self.ffn(self.norm2(tokens), height, width)

        return tokens_to_feature_map(tokens, height, width)


class CrossScaleCNN(nn.Module):
    """Align and fuse two scales with depthwise-pointwise convolutions."""

    def __init__(
        self,
        channels: int = 40,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("channels and hidden_channels must be positive")

        fused_channels = channels * 2
        self.depthwise1 = nn.Conv2d(
            fused_channels,
            fused_channels,
            kernel_size=3,
            padding=1,
            groups=fused_channels,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(fused_channels)
        self.pointwise1 = nn.Conv2d(
            fused_channels,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.depthwise2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.norm3 = nn.BatchNorm2d(hidden_channels)
        self.pointwise2 = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.norm4 = nn.BatchNorm2d(channels)

        self.depthwise3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm5 = nn.BatchNorm2d(channels)
        self.activation = nn.Hardswish(inplace=True)

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
    ) -> torch.Tensor:
        high_feature = resize_to(high_feature, low_feature.shape[-2:])
        feature = torch.cat((low_feature, high_feature), dim=1)

        feature = self.activation(self.norm1(self.depthwise1(feature)))
        feature = self.activation(self.norm2(self.pointwise1(feature)))
        feature = self.activation(self.norm3(self.depthwise2(feature)))
        feature = self.activation(self.norm4(self.pointwise2(feature)))
        return self.norm5(self.depthwise3(feature))


class DIMOCompactDecoder(nn.Module):
    """Fuse TinyNeXt features after optional photometric compensation."""

    def __init__(
        self,
        decoder_channels: int = 40,
        encoder_channels: Sequence[int] = (32, 64, 96, 192),
    ) -> None:
        super().__init__()

        if len(encoder_channels) != 4:
            raise ValueError(
                "encoder_channels 必须依次提供 E1/E2/E3/E4 四个通道数，"
                f"当前为 {tuple(encoder_channels)}。"
            )
        e1_channels, e2_channels, e3_channels, e4_channels = (
            int(channels) for channels in encoder_channels
        )
        if min(e1_channels, e2_channels, e3_channels, e4_channels) <= 0:
            raise ValueError("encoder_channels 中的通道数必须全部大于 0。")

        self.project_e1 = ConvBNReLU(
            e1_channels,
            decoder_channels // 2,
        )
        self.project_e2 = ConvBNReLU(
            e2_channels,
            decoder_channels // 2,
        )
        self.project_e3 = ConvBNReLU(e3_channels, decoder_channels)
        self.project_e4 = ConvBNReLU(e4_channels, decoder_channels)
        self.project_global = ConvBNReLU(
            80,
            decoder_channels,
        )

        multiscale_dilations = (1, 2, 4, 6)
        local_dilations = (1, 1, 1, 1)

        self.low_mid_fusion = make_gpc_stack(
            decoder_channels,
            decoder_channels,
            multiscale_dilations,
        )
        self.low_global_fusion = make_gpc_stack(
            decoder_channels * 2,
            decoder_channels,
            multiscale_dilations,
        )
        self.full_fusion = make_gpc_stack(
            decoder_channels * 2,
            decoder_channels,
            local_dilations,
        )

        self.mid_high_fusion = CrossScaleAttention(
            decoder_channels,
            query_residual=False,
        )
        self.mid_global_fusion = CrossScaleCNN(
            channels=decoder_channels, hidden_channels=64,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        compensation: nn.Module = None,
        image: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        if len(features) != 5:
            raise ValueError(
                "解码器需要 [E1, E2, E3, E4, Global] 五层特征，"
                f"当前收到 {len(features)} 层。"
            )

        e1, e2, e3, e4, global_feature = features

        low = self.project_e1(e1)
        low_middle = self.project_e2(e2)
        middle = self.project_e3(e3)
        high = self.project_e4(e4)
        projected_global = self.project_global(global_feature)

        middle_high = self.mid_high_fusion(middle, high)

        low_middle = resize_to(low_middle, low.shape[-2:])
        low_mid_result = self.low_mid_fusion(
            torch.cat((low, low_middle), dim=1)
        )

        middle_global = self.mid_global_fusion(
            middle_high,
            projected_global,
        )

        global_upsampled = resize_to(
            projected_global,
            low_mid_result.shape[-2:],
        )
        if compensation is None:
            local_for_fusion, global_for_fusion = low_mid_result, global_upsampled
        else:
            if image is None:
                raise ValueError("Photometric compensation requires the input image")
            local_for_fusion, global_for_fusion = compensation(
                low_mid_result, global_upsampled, image,
            )
        low_global = self.low_global_fusion(
            torch.cat((local_for_fusion, global_for_fusion), dim=1)
        )

        middle_global_upsampled = resize_to(
            middle_global,
            low_global.shape[-2:],
        )
        full_result = self.full_fusion(
            torch.cat((low_global, middle_global_upsampled), dim=1)
        )

        # Output order matches the six supervision targets in losses.py.
        return [
            full_result,
            low_mid_result,
            global_feature,
            middle_global,
            low_global,
            middle_high,
        ]
