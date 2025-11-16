import torch
import torch.nn as nn

class CRNNFeatureExtractor(nn.Module):
    """
    CNN backbone for CRNN.
    Produces feature map → reshaped into (T, batch, features) for LSTM.
    """

    def __init__(self, img_channels=1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 32x256 → 16x128

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 16x128 → 8x64

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x64 → 4x64

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x64 → 2x64

            nn.Conv2d(512, 512, 2, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Input: (B, C, H, W)
        Output: (W', B, feature_dim)
        """
        conv_output = self.cnn(x)  # (B, 512, 1?, W')
        b, c, h, w = conv_output.size()

        # collapse height dimension
        assert h == 1 or h == 2
        conv_output = conv_output.mean(2)   # (B, 512, W')

        # permute for LSTM
        conv_output = conv_output.permute(2, 0, 1)  # (W', B, 512)

        return conv_output
