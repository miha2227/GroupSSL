import torch.utils.data as data
import dataset.cifar10 as dataset
import torchvision.transforms as transforms
import numpy as np
import random
import torch

def setup_data(args):
    print(f'==> Preparing cifar10')
    transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])

    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10('./data', args.n_labeled,
                                                                                    transform_train=transform_train,
                                                                                    transform_val=transform_val,
                                                                                    n_augment_unlabeled=args.K)
    n_classes = len(np.unique(train_labeled_set.targets))  # number of classes in the dataset
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, n_classes


def get_anchor_and_nonanchor_points(labels, num_points_per_class, num_classes=100):
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


def get_anchor_and_nonanchor_points_for_mixed_continuous_labels(labels, num_points_per_class, num_classes=100):
    """
        Returns the anchor and non-anchor points as required by GTG for mixed labels (as a result of mmixup procedure)

        Inputs:
        'labels' must be a matrix of dimension NxC, where N = Batch Size and C = Number of classes in the dataset.

        Outputs:
        labs: matrix of dimension NxC with label values for anchor points are set, rest are zero
        L: list of row indices of the anchor points
        U: list of row indices of the non-anchor points

        As the labels are mixed so the assumption is there are no crisp labels and so anchor points are not chosen
        based on exact labels, instead they are chosen randomly (at least until we can thik of some better ways)!!!
    """
    labs = torch.zeros_like(labels)
    total_num_points = num_points_per_class * num_classes
    master_list = [i for i in range(labels.shape[0])]
    L = random.sample(master_list, total_num_points)  # anchor points
    U = list(set(master_list)-set(L))  # non anchor points
    labs[L, :] = labels[L, :]

    return labs, L, U