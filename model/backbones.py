import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNNBackbone(nn.Module):
    """Simple CNN backbone for CRNN."""
    def __init__(self, img_channels=1, dropout_rate=0.0):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_rate),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_rate),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 2, stride=1, padding=0),
            nn.ReLU()
        )
        self.output_channels = 512

    def forward(self, x):
        conv_output = self.cnn(x)
        b, c, h, w = conv_output.size()
        assert h == 1 or h == 2
        conv_output = conv_output.mean(2)
        conv_output = conv_output.permute(2, 0, 1)
        return conv_output


class ResNet18Backbone(nn.Module):
    """ResNet18 adapted for CRNN."""
    def __init__(self, img_channels=1, dropout_rate=0.0):
        super().__init__()
        if img_channels == 1:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove avgpool and fc
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.output_channels = 512
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.features(x)  # (B, 512, H', W')
        b, c, h, w = x.size()

        # Handle height
        if h > 1:
            x = x.mean(dim=2, keepdim=True)  # (B, 512, 1, W')

        x = x.squeeze(2)  # (B, 512, W')
        x = self.dropout(x)
        x = x.permute(2, 0, 1)  # (W', B, 512)
        return x


class ResNet34Backbone(nn.Module):
    """ResNet34 adapted for CRNN."""
    def __init__(self, img_channels=1, dropout_rate=0.0):
        super().__init__()
        if img_channels == 1:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.output_channels = 512
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.features(x)
        b, c, h, w = x.size()
        if h > 1:
            x = x.mean(dim=2, keepdim=True)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)
        return x


class VGG11Backbone(nn.Module):
    """VGG11 adapted for CRNN."""
    def __init__(self, img_channels=1, dropout_rate=0.0):
        super().__init__()
        if img_channels == 1:
            vgg = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        else:
            vgg = models.vgg11(weights=models.VGG11_Weights.DEFAULT)

        self.features = vgg.features
        self.output_channels = 512
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.features(x)  # (B, 512, H', W')
        b, c, h, w = x.size()
        if h > 1:
            x = x.mean(dim=2, keepdim=True)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)
        return x


class VGG16Backbone(nn.Module):
    """VGG16 adapted for CRNN."""
    def __init__(self, img_channels=1, dropout_rate=0.0):
        super().__init__()
        if img_channels == 1:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        else:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.features = vgg.features
        self.output_channels = 512
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.features(x)
        b, c, h, w = x.size()
        if h > 1:
            x = x.mean(dim=2, keepdim=True)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)
        return x


def get_backbone(backbone_name, img_channels=1, dropout_rate=0.0):
    """Factory function to get backbone model."""
    backbones = {
        "simple_cnn": SimpleCNNBackbone,
        "resnet18": ResNet18Backbone,
        "resnet34": ResNet34Backbone,
        "vgg11": VGG11Backbone,
        "vgg16": VGG16Backbone,
    }

    if backbone_name not in backbones:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(backbones.keys())}")

    return backbones[backbone_name](img_channels=img_channels, dropout_rate=dropout_rate)
