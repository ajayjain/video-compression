import argparse
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
from torchvision.models import densenet

from image_identity import ImageIdentity
from autoencoder import CompressUncompressDenseNet
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
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-b', default=32, type=int, metavar='N',
                            help='mini-batch size (1 = pure stochastic) Default: 32')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR',
                            help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                            help='momentum to use with NAG SGD')
parser.add_argument('--resume', default='', help='path to checkpoint from which to resume (defualt: none)')
parser.add_argument('--print-freq', default=5, type=int, help='iteration interval at which updates are printed during training')
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
    transforms.RandomSizedCrop(256),
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

# Create a small container to apply DataParallel to the compression-uncompression network
#class DataParallel(nn.Module):
#    def __init__(self, base_model):
#        super(DataParallel, self).__init__(model=base_model)
#
#    def forward(self, input):
#        if False and torch.cuda.device_count() > 1:
#            gpu_ids = range(torch.cuda.device_count())
#            return nn.parallel.data_parallel(self.model, input, gpu_ids)
#        else:
#            return self.model(input.cuda()).cpu()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs('models_dense', exist_ok=True)
    dest = os.path.join('models_dense', filename)
    torch.save(state, dest)
    if is_best:
        shutil.copyfile(dest, os.path.join('models_dense', 'model_best.pth.tar'))


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
        #for (index, module) in model.module.compress.layers._modules.items():
        #    if isinstance(module, nn.modules.conv.Conv2d):
        #        norm = module.weight.pow(2).sum(3).sum(2).sum(1).rsqrt()
        #        norm = norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #        normalized = module.weight * norm
        #        module.weight.data.copy_(normalized.data)

        #for (index, module) in model.module.uncompress.layers._modules.items():
        #    if isinstance(module, nn.modules.conv.Conv2d):
        #        norm = module.weight.pow(2).sum(3).sum(2).sum(0).rsqrt()
        #        norm = norm.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #        normalized = module.weight * norm
        #        module.weight.data.copy_(normalized.data)

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


def train(train_loader, val_loader):
    # Initialize grayscale model
    dense_model = densenet.densenet121(pretrained=True)
    base_model = CompressUncompressDenseNet(dense_model)
    if use_cuda:
        base_model.cuda()
    model = nn.DataParallel(base_model)

    # Define loss function and optimizer
    # TODO(ajayjain): switch to image_compression.RateDistortionLoss
    criterion = nn.MSELoss()
    if use_cuda:
        criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), args.lr)

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

