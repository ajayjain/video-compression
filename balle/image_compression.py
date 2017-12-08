import operator

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


class GDN2d(nn.Module):
    def __init__(self, num_features):
        super(GDN2d, self).__init__()
        self.num_features = num_features

        self.gamma_conv = nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1, stride=1, padding=0)
        self.beta = Parameter(torch.Tensor((num_features, 1, 1)))

        self.reset_parameters()

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

        # Calculate element-wise normalization factors
        u = input.pow(2)
        u = self.gamma_conv(u)
        u = u + self.beta.expand_as(u)
        u = u.rqrt()

        # Normalize input
        return input * u

    def __repr__(self):
        return ('{name}({num_features}'
                .format(name=self.__class__.__name__, **self.__dict__))


class IGDN2d(GDN2d):
    def __init__(self, num_features):
        super(IGDN2d, self).__init__()

    def forward(self, input):
        if self.training:
            size = list(input.size())
            if reduce(operator.mul, size[2:], size[0]) == 1:
                raise ValueError(
                    'Expected more than 1 value per channel when training, got input size {}'\
                    .format(size)
                )

        # Calculate element-wise inverse normalization factors
        w = input.pow(2)
        w = self.gamma_conv(w)
        w = w + self.beta.expand_as(w)
        w = w.sqrt()

        # Apply elementwise factors
        return input * w


class Compress(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
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
            GDN2d(channels),

            # Stage 3
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            GDN2d(channels),
        )

    def forward(self, x):
        return self.layers(x)


class Uncompress(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        self.layers = nn.Sequential(
            # Stage 1
            IGDN2d(channels),             # invert the normalization transform
            nn.Upsample(scale_factor=2),  # nearest-neighbor upsampling
            nn.Conv2d(                    # convolutional filter
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            # Stage 2
            IGDN2d(channels),           
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            # Stage 3
            IGDN2d(channels),
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


def CompressUncompress(nn.Module):
    def __init__(self, image_channels=1, inner_channels=128):
        self.compress =
            Compress(
                in_channels=image_channels,
                out_channels=inner_channels
            )

        self.uncompress =
            Uncompress(
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
            q = y + (torch.rand(y.shape) - 0.5)
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

