# Video compression

Experimental project for neural image and video compression.

## Overview

This project contains three approaches to media compression:

- Non-negative matrix factorization of YUV/YCbCr values for videos, as a baseline
- A partial implementation of [End-to-end optimized image compression Ballé et al., ICLR’17](www.cns.nyu.edu/~lcv/iclr2017/), including GDN/IGDN non-linearities.
- A fully convolutional deep autoencoder based off of a DenseNet architecture for encoding.

The implementation of Ballé et al. as of now lacks an entropy coder or estimation of a piecewise linear PDF for intermediate activations. Instead, it is trained on a mean-squared error loss.

