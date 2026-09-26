"""Granularity supervision for the decoder's six outputs."""
import torch
from torch import nn
class BCEWithDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum((1, 2, 3))
        denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return bce + (1.0 - dice).mean()

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.region = BCEWithDiceLoss()

    def forward(self, outputs, targets):
        z, y = outputs['granularity_logits'].float(), targets.float()
        full = self.region(z[:, 0:1], y[:, 0:1])
        center = self.region(z[:, 3:4], y[:, 2:3])
        peripheral = self.region(z[:, 4:5], y[:, 4:5])
        return full + center + peripheral
