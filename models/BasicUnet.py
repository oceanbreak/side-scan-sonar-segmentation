"""Two-level U-Net baseline with bilinear upsampling."""

import torch
from torch import nn
from torch.nn import functional as F


def _double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class BasicUnet(nn.Module):
    """Return raw segmentation logits at the input resolution (H, W >= 4)."""

    def __init__(self, in_channels=1, num_classes=4, base_channels=16):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel and class counts must be positive")
        b = base_channels
        self.encoder1 = _double_conv(in_channels, b)
        self.encoder2 = _double_conv(b, 2 * b)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _double_conv(2 * b, 4 * b)
        self.decoder2 = _double_conv(6 * b, 2 * b)
        self.decoder1 = _double_conv(3 * b, b)
        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 4:
            raise ValueError("Expected (N, C, H, W) with H and W at least 4")
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        x = self.bottleneck(self.pool(skip2))
        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder2(torch.cat([x, skip2], dim=1))
        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder1(torch.cat([x, skip1], dim=1))
        return self.head(x)
