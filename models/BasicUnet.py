"""Small U-Net accepting arbitrary patch dimensions >= 4."""
import torch
from torch import nn
from torch.nn import functional as F


def _block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class BasicUNet(nn.Module):
    def __init__(self, num_classes=4, width=16):
        super().__init__()
        self.encoder1 = _block(1, width)
        self.encoder2 = _block(width, width * 2)
        self.bottleneck = _block(width * 2, width * 4)
        self.decoder2 = _block(width * 6, width * 2)
        self.decoder1 = _block(width * 3, width)
        self.output = nn.Conv2d(width, num_classes, 1)

    def forward(self, x):
        first = self.encoder1(x)
        second = self.encoder2(F.max_pool2d(first, 2))
        center = self.bottleneck(F.max_pool2d(second, 2))
        up = F.interpolate(center, size=second.shape[-2:], mode='bilinear', align_corners=False)
        up = self.decoder2(torch.cat([up, second], dim=1))
        up = F.interpolate(up, size=first.shape[-2:], mode='bilinear', align_corners=False)
        return self.output(self.decoder1(torch.cat([up, first], dim=1)))
