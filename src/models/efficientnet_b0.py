# src/models/efficientnet_b0.py

###
### EfficientNet-B0 implementation (adapted for TinyImageNet 64x64)
###

# import
import torch
import torch.nn as nn
import torch.nn.functional as F

# implementation

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        hidden_dim = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# MBConv block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, kernel_size=3, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = (in_channels == out_channels and stride == 1)

        layers = []
        # expansion (1x1 conv)
        if expansion != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU(inplace=True))

        # depthwise conv
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False)
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU(inplace=True))

        # squeeze-and-excitation
        if se_ratio is not None:
            layers.append(SEBlock(hidden_dim, reduction=int(1/se_ratio)))

        # projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out
    
# EfficientNet-B0 (for imagenet)
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=200):
        super(EfficientNetB0, self).__init__()

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # EfficientNet-B0 config
        # (expansion, out_channels, num_blocks, stride, kernel_size)
        config = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        layers = []
        in_channels = 32
        for expansion, out_channels, num_blocks, stride, k in config:
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(MBConvBlock(in_channels, out_channels, expansion, s, k))
                in_channels = out_channels

        self.blocks = nn.Sequential(*layers)

        # head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.classifier(x)
        return x