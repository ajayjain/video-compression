import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from image_identity import ImageIdentity
from image_compression import CompressUncompress
from grayscale_transform import Grayscale


parser = argparse.ArgumentParser(description='Training an End-to-End image compression model')
# Data arguments
parser.add_argument('--train-dir', metavar='PATH', required=True,
                            help='path to train data folder')
parser.add_argument('--val-dir', metavar='PATH', required=True,
                            help='path to validation data folder')
parser.add_argument('--color-mode', default='gray', help='input/output color mode [gray | color] (default: gray)')
parser.add_argument('--threads', '-j', default=2, type=int, metavar='N',
                            help='number of data loading threads (default: 2)')
# Training arguments
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
parser.add_argument('--epoch-number', default=1, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-b', default=256, type=int, metavar='N',
                            help='mini-batch size (1 = pure stochastic) Default: 256')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR',
                            help='initial learning rate')
# parser.add_argument('--resume-checkpoint', default='', help='path to checkpoint from which to resume (defualt: none)')
args = parser.parse_args()


# TODO(ajayjain): consider setting cudnn.benchmark = True, if input size is fixed.
# cudnn.benchmark = True


# Data loading
# TODO(ajayjain): Balle et al. removed images with "excessive saturation", and
#                 added uniform noise to pixel values
if args.color_mode == 'gray':
    transform_list = [Grayscale()]
else:
    transform_list = []
transform_list.extend([
    transforms.RandomSizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform = transforms.Compose(transform_list)

train = ImageIdentity(args.train_dir, transform)
val = ImageIdentity(args.val_dir, transform)
train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.threads
)


# Create a small container to apply DataParallel to the compression-uncompression network
class DataParallel(nn.Container):
    def __init__(self, base_model):
        super(DataParallel, self).__init__(model=base_model)

    def forward(self, input):
        if torch.cuda.device_count() > 1:
            gpu_ids = range(torch.cuda.device_count())
            return nn.parallel.data_parallel(self.model, input, gpu_ids)
        else:
            return self.model(input.cuda()).cpu()


# Initialize grayscale model
base_model = CompressUncompress(image_channels=1)
model = DataParallel(base_model)


# Define loss function and optimizer
# TODO(ajayjain): switch to image_compression.RateDistortionLoss
loss = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), args.lr)


# Pass model, loss, optimizer, and data loader to the trainer
t = trainer.Trainer(model, loss, optimizer, train_loader)


# Register some monitoring plugins
# TODO(ajayjain): Monitor bitrate and reconstruction loss components separately
t.register_plugin(trainer.plugins.ProgressMonitor())
t.register_plugin(trainer.plugins.LossMonitor())
t.register_plugin(trainer.plugins.TimeMonitor())
t.register_plugin(trainer.plugins.Logger(['progress', 'loss', 'time']))


# Start training!
t.run(args.epochs)

