# src/models/shufflenetv2.py

###

### ShuffleNetV2 model implementation

###

# imports

import torch
import torch.nn as nn
import torch.nn.functional as F

# helper block: depthwise separable convolution

def conv_1x1(in_channels, out_channels):
    return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    )

def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
    nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
    nn.BatchNorm2d(in_channels),
    nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    )

# ShuffleNetV2 basic unit

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if stride == 1:
            # equal split
            self.branch = nn.Sequential(
                conv_1x1(mid_channels, mid_channels),
                conv_dw(mid_channels, mid_channels, stride=1),
                conv_1x1(mid_channels, mid_channels),
            )
        else:
            # projection branch
            self.branch_proj = nn.Sequential(
                conv_dw(in_channels, in_channels, stride),
                conv_1x1(in_channels, mid_channels),
            )
            self.branch = nn.Sequential(
                conv_1x1(in_channels, mid_channels),
                conv_dw(mid_channels, mid_channels, stride),
                conv_1x1(mid_channels, mid_channels),
            )

    def channel_shuffle(self, x, groups=2):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        if self.stride == 1:
            c = x.shape[1] // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out2 = self.branch(x2)
            out = torch.cat((x1, out2), 1)
        else:
            out1 = self.branch_proj(x)
            out2 = self.branch(x)
            out = torch.cat((out1, out2), 1)

        out = self.channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=200):
        super(ShuffleNetV2, self).__init__()

        # initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # stages (using ShuffleNetV2 0.5x config for simplicity)
        self.stage2 = self._make_stage(24, 48, 4)
        self.stage3 = self._make_stage(48, 96, 8)
        self.stage4 = self._make_stage(96, 192, 4)

        out_channels = 192

        # final conv
        self.conv5 = conv_1x1(out_channels, 1024)

        # classifier
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, stride=2))
        for i in range(repeat - 1):
            layers.append(ShuffleUnit(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
