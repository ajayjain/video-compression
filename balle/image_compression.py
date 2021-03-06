from functools import reduce
import math
import operator

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn


class GDN2d(nn.Module):
    def __init__(self, num_features, eps=1e-9):
        super(GDN2d, self).__init__()
        self.num_features = num_features

        self.conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Average the weight gradient with its transpose to make the
        # weight matrix symmetric (wrt input and output channels)
        def average_with_transpose(grad):
            return 0.5 * (grad + grad.transpose(0, 1))
        self.conv.weight.register_hook(average_with_transpose)

        self.eps = eps

    def forward(self, input):
        if self.training:
            # Error check borrowed from PyTorch's nn.functional.batch_norm implementation
            size = list(input.size())
            if reduce(operator.mul, size[2:], size[0]) == 1:
                raise ValueError(
                    'Expected more than 1 value per channel when training, got input size {}'\
                    .format(size)
                )

        # Calculate element-wise normalization factors
        u = input.pow(2)
        u = self.conv(u)
        # Apply relu to u, so the sqrt argument is nonnegative
        # and add eps to have a nonzero denominator
        u = nn.functional.relu(u) + self.eps
        u = u.rsqrt()

        # Normalize input
        return input * u

    def __repr__(self):
        return ('{name}({num_features}'
                .format(name=self.__class__.__name__, **self.__dict__))


class IGDN2d(GDN2d):
    def __init__(self, num_features):
        super(IGDN2d, self).__init__(num_features)

    def forward(self, input):
        if self.training:
            size = list(input.size())
            if reduce(operator.mul, size[2:], size[0]) == 1:
                raise ValueError(
                    'Expected more than 1 value per channel when training, got input size {}'\
                    .format(size)
                )

        return nn.functional.tanh(input)

        # Calculate element-wise inverse normalization factors
        w = torch.pow(input, 2)
        w = self.conv(w)
        # Apply relu to w, so sqrt arg is nonnegative
        w = nn.functional.relu(w)
        w = w.sqrt()

        # Apply elementwise factors
        return input * w


class Compress(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(Compress, self).__init__()
        # NOTE(ajayjain): I'm guessing the padding, but it looks like the 
        # authors simply pad the image once at the beginning.
        # Downsampling is implemented via strided convolutions

        # Let the number of intermediate channels equal the output channels
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(4 + 4*2 + 4*2*2),

            # Stage 1
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=9,
                stride=4,
                padding=0
            ),
            GDN2d(out_channels),

            # Stage 2
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=2,
                padding=0
            ),
            GDN2d(out_channels),

            # Stage 3
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=2,
                padding=0
            ),
            GDN2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Uncompress(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        super(Uncompress, self).__init__()
        self.layers = nn.Sequential(
            # Stage 1
            IGDN2d(in_channels),          # invert normalization transform
            nn.Upsample(scale_factor=2),  # nearest-neighbor upsampling
            nn.Conv2d(                    # convolutional filter
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            # Stage 2
            IGDN2d(in_channels),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            # Stage 3
            IGDN2d(in_channels),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=9,
                stride=1,
                padding=4
            ),
        )

    def forward(self, x):
        reconstructed = self.layers(x)
        if not self.training:
            reconstructed = torch.clamp(reconstructed, min=0.0, max=1.0)
        return reconstructed


class CompressUncompress(nn.Module):
    def __init__(self, image_channels=1, inner_channels=256, pretrained=True):
        super(CompressUncompress, self).__init__()

        self.pretrained = pretrained

        self.compress = Compress(
            in_channels=image_channels,
            out_channels=inner_channels
        )

        self.uncompress = Uncompress(
            in_channels=inner_channels,
            out_channels=image_channels
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TODO(ajayjain): Look into other initializations
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # TODO(ajayjain): parameters for spline approximating p_{y_i} (y_i)

    def forward(self, input):
        # Transform input to latent code space, y = g_a(x; phi)
        y = self.compress(input)

        # (Relaxed) quantization step for transmission
        if self.training:
            # Relaxed quantization:
            #   Add noise, sampled uniformly on [-0.5, 0.5)
            noise = torch.autograd.Variable(torch.rand(y.size()).cuda())
            noise -= 0.5
            q = y + noise
        else:
            # Quantization by rounding
            # TODO(ajayjain): Verify this works, as torch.round may not
            # be available for variables
            q = torch.round(y)

        # Reinterpret q as an approximation of the pre-transmission code:
        #   y_hat = q,
        # and transform back to the data space:
        #   x_hat = g_s(y_hat; theta) = g_s(q; theta)
        reconstructed = self.uncompress(q)

        return reconstructed


class RateDistortionLoss(nn.Module):
    def __init__(self, lamb):
        self.lamb = lamb
        super(RateDistortionLoss, self).__init__()

        self.distortion_loss = nn.MSELoss()

    def forward(self, original, reconstructed):
        D = self.distortion_loss(original, reconstructed)
        # TODO(ajayjain): Implement rate estimate
        R = 0

        return R + self.lamb * D

