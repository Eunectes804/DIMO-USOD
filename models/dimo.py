"""DIMO-USOD network with checkpoint-compatible parameter names."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .decoder import (DIMOCompactDecoder, GlobalTransformerBlock,
                      feature_map_to_tokens, tokens_to_feature_map, resize_to)
from .backbone import TinyNeXtSBackbone

def cpcs(probability, alpha=2.2):
    """Parameter-free confidence sharpening on the model input grid."""
    if not math.isfinite(alpha) or alpha < 1:
        raise ValueError('CPCS alpha must be finite and >= 1')
    if alpha == 1:
        return probability
    return torch.sigmoid(torch.logit(probability.float()) * alpha)


def conv_block(ci, co, kernel=3, dilation=1, groups=8):
    return nn.Sequential(
        nn.Conv2d(ci, co, kernel, padding=(kernel//2)*dilation,
                  dilation=dilation, bias=False),
        nn.GroupNorm(groups, co), nn.SiLU())


def expert(ci, dilations):
    return nn.Sequential(conv_block(ci, 64, 1),
        conv_block(64, 64, 3, dilations[0]),
        conv_block(64, 64, 3, dilations[1]),
        nn.Conv2d(64, 40, 1, bias=True))


def recover_rgb(image):
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    return (image.float()*std+mean).clamp(0, 1)


def photometric_descriptor(image, output_size):
    """Compute RGB luma, local mean, and local contrast before downsampling.

    The descriptors summarize observed appearance, not physical illumination.
    """
    rgb = recover_rgb(image)
    weights = rgb.new_tensor([.2126, .7152, .0722])[None, :, None, None]
    y = (rgb*weights).sum(1, keepdim=True)
    radius = max(1, round(4*min(image.shape[-2:])/320))
    kernel = radius*2+1
    mean = F.avg_pool2d(F.pad(y, (radius,)*4, mode='reflect'), kernel, stride=1)
    second = F.avg_pool2d(F.pad(y.square(), (radius,)*4, mode='reflect'), kernel, stride=1)
    contrast = (second-mean.square()).clamp_min(0).add(1e-6).sqrt()
    return F.interpolate(torch.cat((y, mean, contrast), 1), size=output_size, mode='area')


class Compensation(nn.Module):
    """Photometric spatial logits + visual channel context; signed residuals."""
    def __init__(self):
        super().__init__()
        self.local = expert(40, (1, 1))
        self.context = expert(80, (2, 4))
        self.condition_encoder = nn.Sequential(conv_block(3, 16, 3, groups=4),
                                               conv_block(16, 16, 3, groups=4))
        self.condition_encoder[0][1] = nn.Identity()
        self.condition_encoder[1][1] = nn.Identity()
        self.register_buffer('fixed_mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('fixed_scale', torch.ones(1, 3, 1, 1))
        self.spatial_gate = nn.Sequential(conv_block(16, 32, 1, groups=4),
            conv_block(32, 32, 3, groups=4), nn.Conv2d(32, 40, 1, bias=True))
        self.global_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(80, 16, 1), nn.SiLU(), nn.Conv2d(16, 40, 1))
        for branch in (self.spatial_gate, self.global_gate, self.local, self.context):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        self.capture_maps = False
        self.last_maps = None

    def set_fixed_statistics(self, record):
        mean = torch.as_tensor(record['mean'], dtype=torch.float32, device=self.fixed_mean.device).view_as(self.fixed_mean)
        scale = torch.as_tensor(record['scale'], dtype=torch.float32, device=self.fixed_scale.device).view_as(self.fixed_scale)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid fixed photometric statistics')
        with torch.no_grad():
            self.fixed_mean.copy_(mean)
            self.fixed_scale.copy_(scale)

    def forward(self, local, global_feature, image):
        joint = torch.cat((local, global_feature), 1)
        rl, rg = self.local(local), self.context(joint)
        descriptor = photometric_descriptor(image, local.shape[-2:])
        condition = self.condition_encoder((descriptor - self.fixed_mean) / self.fixed_scale)
        allocation = (self.spatial_gate(condition) + self.global_gate(joint)).sigmoid()
        dl, dg = 2 * allocation * rl, 2 * (1 - allocation) * rg
        if self.capture_maps:
            cl, cg = dl.detach().abs().mean(1, keepdim=True), dg.detach().abs().mean(1, keepdim=True)
            self.last_maps = {'gate': allocation.detach().mean(1, keepdim=True),
                              'local_correction': cl, 'context_correction': cg,
                              'local_fraction': cl / (cl + cg + 1e-6)}
        return local + dl, global_feature + dg


class DIMOUSOD(nn.Module):
    """DIMO-USOD with optional compensation for the Base ablation.

    forward(): training dictionary of raw logits at input resolution.
    predict(): deployed continuous saliency probabilities, CPCS alpha=2.2.
    Inputs use ImageNet RGB normalization. GT/depth are never inference inputs.
    """
    def __init__(self, pretrained_backbone_path=None, compensation=True, module_seed=622029):
        super().__init__()
        self.return_all = True
        self.backbone = TinyNeXtSBackbone(pretrained_backbone_path)
        self.global_blocks = nn.ModuleList([
            GlobalTransformerBlock(192, 80, num_heads=4),
            GlobalTransformerBlock(80, 80, num_heads=4),
        ])
        self.decoder = DIMOCompactDecoder(
            decoder_channels=40,
            encoder_channels=TinyNeXtSBackbone.OUTPUT_CHANNELS,
        )
        self.prediction_heads = nn.ModuleList([
            nn.Conv2d(channels, 1, 1)
            for channels in (40, 40, 80, 40, 40, 40)
        ])
        self.use_compensation = bool(compensation)
        if compensation:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(module_seed)
                self.compensation = Compensation()

    def decode_features(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected RGB tensor [B, 3, H, W], got {tuple(image.shape)}")
        enc = self.backbone(image)
        h, w = enc[-1].shape[-2:]
        tokens = feature_map_to_tokens(enc[-1])
        for block in self.global_blocks:
            tokens = block(tokens, h, w)
        global_feature = tokens_to_feature_map(tokens, h, w)
        features = self.decoder(
            [*enc, global_feature],
            compensation=self.compensation if self.use_compensation else None,
            image=image,
        )
        return enc, features

    def forward(self, image):
        _, features = self.decode_features(image)
        native_logits = self.prediction_heads[0](features[0])
        final_logits = resize_to(native_logits, image.shape[-2:])
        if not self.return_all:
            return final_logits.sigmoid()
        granularity_logits = torch.cat([
            resize_to(native_logits if index == 0 else head(feature), image.shape[-2:])
            for index, (head, feature) in enumerate(zip(self.prediction_heads, features))
        ], dim=1)
        return {
            "final_logits": final_logits,
            "granularity_logits": granularity_logits,
            "detail_feature": features[1],
        }

    @torch.no_grad()
    def predict(self, image, alpha=2.2):
        _, features = self.decode_features(image)
        logits = resize_to(self.prediction_heads[0](features[0]), image.shape[-2:])
        return cpcs(logits.float().sigmoid(), alpha)
