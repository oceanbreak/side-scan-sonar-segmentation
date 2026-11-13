"""Small fully convolutional baseline for full-width sonar strips."""
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, images):
        return self.layers(images)
