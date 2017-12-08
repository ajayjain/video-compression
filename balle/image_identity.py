import os

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.datasets.folder import is_image_file, default_loader, IMG_EXTENSIONS


def find_images(directory):
    """Recursively search a directory for images, returning a list of paths"""
    paths = []

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                paths.append(path)

    return paths


class ImageIdentity(data.Dataset):
    """A generic data loader where the images are arranged in this way:
        root/xxx.png
        root/xxy.png
        root/xxz.png
        root/123.png
        root/nsdf3.png
        root/asd932_.png

       Target tensors are equal to the loaded tensors, if transformations are not supplied.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. The default target is the same as the transformed
            input.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        imgs = find_images(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image
        """
        path = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        target = img
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

