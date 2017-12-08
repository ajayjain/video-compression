# Implemented here, as my version of torchvision does not have the class
# Taken from https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py

from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.
    Args:
        img (PIL Image): Image to be converted to grayscale.
    Returns:
        PIL Image:  Grayscale version of the image.
                    if num_output_channels == 1 : returned image is single channel
                    if num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


class Grayscale(object):
    """Convert image to grayscale.
    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b
    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.
        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return to_grayscale(img, num_output_channels=self.num_output_channels)
