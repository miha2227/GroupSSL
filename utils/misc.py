'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import torch
import errno
import os
import numpy as np
import torch.backends.cudnn as cudnn
import random
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_labeled_and_unlabeled_points(labels, num_points_per_class, num_classes=100):  # there is a duplicate of this method in data_util with name get_anchor_and_nonanchor_points
    labs, L, U = [], [], []
    labs_buffer = np.zeros(num_classes)
    num_points = labels.shape[0]
    for i in range(num_points):
        if labs_buffer[labels[i]] == num_points_per_class:
            U.append(i)
        else:
            L.append(i)
            labs.append(labels[i])
            labs_buffer[labels[i]] += 1
    return labs, L, U


def get_anchor_and_nonanchor_points_for_unlabeled_data(initial_guessed_labels, num_points_per_class, num_classes=10, n_augment=2, batch_size=64):  # there is a duplicate of this method in data_util with name get_anchor_and_nonanchor_points
    labs, L, U = [], [], []
    num_points = initial_guessed_labels.shape[0]
    assert num_points == batch_size * n_augment, 'total number of points is not matching with batch size times number of augmentations'
    for j in range(n_augment):
        start = j * batch_size
        stop = start + batch_size
        labs_buffer = np.zeros(num_classes)
        for i in range(start, stop):
            if labs_buffer[initial_guessed_labels[i]] == num_points_per_class:
                U.append(i)
            else:
                L.append(i)
                labs.append(initial_guessed_labels[i])
                labs_buffer[initial_guessed_labels[i]] += 1
    return labs, L, U


def get_labeled_and_unlabeled_points_random_order(labels, num_points_per_class, num_classes=100):
    L, U = [], []
    labs_buffer = np.zeros(num_classes)
    num_points = labels.shape[0]
    idx_list = [j for j in range(num_points)]
    random.shuffle(idx_list)
    for i in idx_list:
        if labs_buffer[labels[i]] == num_points_per_class:
            U.append(i)
        else:
            L.append(i)
            labs_buffer[labels[i]] += 1
    return L, U

def setup_device():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = 'cuda:0'
        cudnn.benchmark = True
    else:
        device = 'cpu'
        cudnn.benchmark = False
    return device, use_cuda


def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def unset_random_seed():
    seed = None
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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