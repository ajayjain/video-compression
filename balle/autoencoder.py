from functools import reduce
import math
import operator

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torchvision.models import densenet


class CompressDenseNet(nn.Module):
    def __init__(self, densemodel):
        super(CompressDenseNet, self).__init__()

        self.model = densemodel
        del self.model.classifier
    
    def forward(self, x):
        return self.model.features(x)


class UncompressDenseNet(nn.Module):
    def __init__(self):
        super(UncompressDenseNet, self).__init__()

        self.layers = nn.Sequential(
            # in: Nx1024x8x8
            nn.Upsample(scale_factor=4),
            nn.Conv2d(
                in_channels=1024,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),

            # in: Nx512x32x32
            nn.Upsample(scale_factor=4),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),

            # in: Nx256x128x128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.ReLU(inplace=True),

            # in: Nx128x256x256
            nn.Conv2d(
                in_channels=128,
                out_channels=3,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class CompressUncompressDenseNet(nn.Module):
    def __init__(self, densemodel):
        super(CompressUncompressDenseNet, self).__init__()

        self.compress = CompressDenseNet(densemodel)
        self.uncompress = UncompressDenseNet()


    def forward(self, x):
        y = self.compress(x)
        # TODO(ajayjain): Add noise to y
        reconstructed = self.uncompress(y)

        return reconstructed

