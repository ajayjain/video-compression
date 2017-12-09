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

        self.gamma_conv = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1, stride=1, padding=0)
        self.beta = nn.Parameter(torch.rand((num_features, 1, 1)) + eps)

        self.reset_parameters()

        self.eps = eps

    def reset_parameters(self):
        # TODO(ajayjain): Is this the best initialization of the parameters? perhaps zero_() for beta?
        self.beta.data.uniform_()

    def forward(self, input):
        if self.training:
            # Error check borrowed from PyTorch's nn.functional.batch_norm implementation
            size = list(input.size())
            if reduce(operator.mul, size[2:], size[0]) == 1:
                raise ValueError(
                    'Expected more than 1 value per channel when training, got input size {}'\
                    .format(size)
                )

        # NOTE(ajayjain): Temporarily using a tanh activation until I can debug
        # explosion of activations with GDN/IGDN
        #nn.functional.softmin(input, dim=
        return nn.functional.tanh(input)

        # Calculate element-wise normalization factors
        u = input.pow(2)
        u = self.gamma_conv(u)
        u = u + self.beta.expand_as(u)
        # Apply relu to u, so sqrt arg is nonnegative
        # and add eps for nonzero denominator
        u = nn.functional.relu(u) + self.eps
        u = u.rsqrt()

        # Normalize input
        return torch.clamp(input * u, max=1e3)

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
        w = torch.clamp(w, min=0, max=1e6)
        print(w)
        w = self.gamma_conv(w)
        w = w + self.beta.expand_as(w)
        # Apply relu to w, so sqrt arg is nonnegative
        w = nn.functional.relu(w)
        w = w.sqrt()

        # Apply elementwise factors
        return input * w


class Compress(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super(Compress, self).__init__()
        # NOTE(ajayjain): I'm guessing the padding, but it looks like the 
        # authors simply pad the image once at the beginning.
        # Downsampling is implemented via strided convolutions

        # Let the number of intermediate channels equal the output channels
        self.layers = nn.Sequential(
            # Stage 1
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=9,
                stride=4,
                padding=4
            ),
            GDN2d(out_channels),

            # Stage 2
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            GDN2d(out_channels),

            # Stage 3
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            GDN2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Uncompress(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super(Uncompress, self).__init__()
        self.layers = nn.Sequential(
            # Stage 1
            # IGDN2d(in_channels),          # invert the normalization transform
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
        return self.layers(x)


class CompressUncompress(nn.Module):
    def __init__(self, image_channels=1, inner_channels=128):
        super(CompressUncompress, self).__init__()

        self.compress = Compress(
            in_channels=image_channels,
            out_channels=inner_channels
        )

        self.noise = torch.autograd.Variable(torch.rand(1), requires_grad = False).cuda()

        self.uncompress = Uncompress(
            in_channels=inner_channels,
            out_channels=image_channels
        )

        # TODO(ajayjain): parameters for spline approximating p_{y_i} (y_i)

    def forward(self, input):
        # Transform input to latent code space, y = g_a(x; phi)
        y = self.compress(input)

        # (Relaxed) quantization step for transmission
        if self.training:
            # Relaxed quantization:
            #   Add noise, sampled uniformly on [-0.5, 0.5)
            #self.noise = self.noise.expand_as(y)
            #self.noise.data.uniform_()
            #self.noise = self.noise - 0.5
            #q = y + self.noise
            q = y
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
    def __init__(self, gamma):
        self.gamma = gamma
        super(RateDistortionLoss, self).__init__()

        self.distortion_loss = nn.MSELoss()

    def forward(self, original, reconstructed):
        D = self.distortion_loss(original, reconstructed)
        # TODO(ajayjain): Implement rate estimate
        R = 0

        return R + self.gamma * D

