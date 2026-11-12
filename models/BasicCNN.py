"""Small fully convolutional baseline for semantic segmentation."""

from torch import nn


class BasicCNN(nn.Module):
    """Return raw (N, num_classes, H, W) logits; use CrossEntropyLoss."""

    def __init__(self, in_channels=1, num_classes=4, base_channels=16):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel and class counts must be positive")
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1),
        )

    def forward(self, x):
        return self.network(x)
