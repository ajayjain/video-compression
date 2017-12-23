import argparse
import math
import os
import shutil
import time

import torch
import torch.autograd as autograd
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
from image_compression import CompressUncompress, RateDistortionLoss
from grayscale_transform import Grayscale


parser = argparse.ArgumentParser(description='Training an End-to-End image compression model')
# Data arguments
parser.add_argument('--train-dir', metavar='PATH', required=True,
                            help='path to train data folder')
parser.add_argument('--val-dir', metavar='PATH', required=True,
                            help='path to validation data folder')
parser.add_argument('--color-mode', default='gray', help='input/output color mode [gray | color] (default: gray)')
parser.add_argument('--binarization', default=0.2, type=float,
                            help='weight of reconstructed tensor binarizing loss')
parser.add_argument('--threads', '-j', default=2, type=int, metavar='N',
                            help='number of data loading threads (default: 2)')
# Training arguments
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-b', default=32, type=int, metavar='N',
                            help='mini-batch size (1 = pure stochastic) Default: 32')
parser.add_argument('--optimizer', default='adam', type=str,
                            help='Optimizer to use [adam | sgd] (default: adam)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR',
                            help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                            help='momentum to use with NAG SGD')
parser.add_argument('--resume', default='',
                            help='path to checkpoint from which to resume (defualt: none)')
parser.add_argument('--print-freq', default=5, type=int,
                            help='iteration interval at which updates are printed during training')
parser.add_argument('--no-parallel', default=False, action='store_true',
                            help='disable DataParallel for single-GPU training')
# Model arguments
parser.add_argument('--inner-channels', default=128, type=int,
                            help='number of intermediate channels')
args = parser.parse_args()


# TODO(ajayjain): consider setting cudnn.benchmark = True, if input size is fixed.
# cudnn.benchmark = True
use_cuda = torch.cuda.is_available()


# Data loading
# TODO(ajayjain): Balle et al. removed images with "excessive saturation", and
#                 added uniform noise to pixel values
if args.color_mode == 'gray':
    transform_list = [Grayscale()]
else:
    transform_list = []
transform_list.extend([
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform = transforms.Compose(transform_list)

train = ImageIdentity(args.train_dir, transform)
train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.threads,
    pin_memory=use_cuda
)

val = ImageIdentity(args.val_dir, transform)
val_loader = torch.utils.data.DataLoader(
    val,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.threads,
    pin_memory=use_cuda
)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs('models', exist_ok=True)
    dest = os.path.join('models', filename)
    torch.save(state, dest)
    if is_best:
        shutil.copyfile(dest, os.path.join('models', 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(model, criterion, loader):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    for i, data in enumerate(loader):
        iter_start_time = time.time()

        batch_input, batch_target = data

        input_var = autograd.Variable(batch_input)
        target_var = autograd.Variable(batch_target)
        if use_cuda:
            input_var = input_var.cuda(async=True)
            target_var = target_var.cuda(async=True)

        batch_output = model(input_var)

        loss = criterion(batch_output, target_var)
        losses.update(loss.data[0], batch_input.size(0))

        batch_time.update(time.time() - iter_start_time)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
                  i, len(loader), batch_time=batch_time,
                  loss=losses))

    print('##############################################################################\n'
          '  Test summary:\n'
          '  loss: {loss.avg:.6f}    time: {total_time}\n'
          '##############################################################################'
          .format(loss=losses, total_time=len(loader)*batch_time.avg))

    return losses.avg


def train_epoch(model, criterion, optimizer, loader, epoch):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    for i, data in enumerate(loader):
        iter_start_time = time.time()

        batch_input, batch_target = data

        input_var = autograd.Variable(batch_input)
        target_var = autograd.Variable(batch_target)
        if use_cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        batch_output = model(input_var)

        loss = criterion(batch_output, target_var)
        losses.update(loss.data[0], batch_input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Post-process weights
        if args.no_parallel:
            modules = model
        else:
            modules = model.module

        for (index, module) in modules.compress.layers._modules.items():
            if isinstance(module, nn.modules.conv.Conv2d):
                norm = module.weight.pow(2).sum(3).sum(2).sum(1).rsqrt()
                norm = norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                normalized = module.weight * norm
                module.weight.data.copy_(normalized.data)

        for (index, module) in modules.uncompress.layers._modules.items():
            if isinstance(module, nn.modules.conv.Conv2d):
                norm = module.weight.pow(2).sum(3).sum(2).sum(0).rsqrt()
                norm = norm.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                normalized = module.weight * norm
                module.weight.data.copy_(normalized.data)

        batch_time.update(time.time() - iter_start_time)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
        	  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        	  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'.format(
        	   epoch, i, len(loader), batch_time=batch_time,
        	   loss=losses))

    print('##############################################################################\n'
          '  Epoch {epoch} summary:\n'
          '  loss: {loss.avg:.6f}    time: {total_time}\n'
          '##############################################################################'
          .format(epoch=epoch, loss=losses, total_time=len(loader)*batch_time.avg))

    return losses.avg


#class BinarizingLoss(nn.Module):
#    def __init__(self):
#        super(BinarizingLoss, self).__init__()
#
#    def forward(self, x):
#        clamped = torch.clamp(x, min=0., max=1.)
#        central_tendancy = 0.5 * torch.pow(torch.sin(clamped * math.pi), 4)
#        mean = central_tendancy.mean()
#        return mean
#
#
#class BinarizingSmoothL1Loss(nn.Module):
#    def __init__(self, binarization_weight=0.2):
#        super(BinarizingSmoothL1Loss, self).__init__()
#
#        self.error_loss = nn.SmoothL1Loss()
#        self.binarizing_loss = BinarizingLoss()
#        self.binarization_weight = binarization_weight
#
#    def forward(self, output, target):
#        error = self.error_loss(output, target)
#        central_tendancy = self.binarizing_loss(output)
#
#        # clamping = torch.nn.functional.relu(output - 1) + torch.nn.functional.relu(-output)
#
#        return ((1 - self.binarization_weight) * error +
#                self.binarization_weight * central_tendancy)


def train(train_loader, val_loader):
    if args.color_mode == 'gray':
        # Initialize grayscale model
        base_model = CompressUncompress(image_channels=1, inner_channels=args.inner_channels)
    else:
        # Initialize 3-channel input model
        base_model = CompressUncompress(image_channels=3, inner_channels=args.inner_channels)

    if use_cuda:
        base_model.cuda()

    if args.no_parallel:
        model = base_model
    else:
        model = nn.DataParallel(base_model)

    # Define loss function and optimizer
    #criterion = BinarizingSmoothL1Loss(args.binarization)
    criterion = RateDistortionLoss(lamb=1)
    if use_cuda:
        criterion.cuda()

    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif args.optimzer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    best_val_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, criterion, optimizer, train_loader, epoch)
        val_loss = validate(model, criterion, val_loader)

        is_best = val_loss < best_val_loss
        best_val_loss = max(best_val_loss, val_loss)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

train(train_loader, val_loader)

