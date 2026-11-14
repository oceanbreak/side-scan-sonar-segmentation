"""A fully convolutional four-class baseline for grayscale sonar patches."""
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self, num_classes=4, width=24):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width * 2, num_classes, 1),
        )

    def forward(self, x):
        return self.layers(x)
