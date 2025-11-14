import torch
import torch.nn as nn

class CRNN(nn.Module):
    """Minimal CRNN backbone stub. Replace conv layers with a real implementation."""
    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)
