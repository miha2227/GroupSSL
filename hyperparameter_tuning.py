import random
from math import log10
from itertools import product
from solver import Solver
import dataset.cifar10 as dataset
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import models.wideresnet as models
from group_loss.gtg import GTG
import torch.nn as nn
import torch.optim as optim
from train import SemiLoss, WeightEMA


ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def random_search(train_loader, val_loader, model_class,
                  random_search_spaces={
                      "T_softmax": ([10.0, 20.0], "float"),
                      "num_labeled_per_class": ([1, 5], "int"),
                      "alpha": ([0.0, 1.0], "float"),
                      "lambda_u": ([70.0, 80.0], "float"),
                      "ema_decay": ([0.8, 1.0], "float"),
                  },
                  num_search=20, epochs=20,
                  patience=5):
    """
    Samples N_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.
    See the grid search documentation above.
    Additional/different optional arguments:
        - random_search_spaces: similar to grid search but values are of the
        form
        (<list of values>, <mode as specified in ALLOWED_RANDOM_SEARCH_PARAMS>)
        - num_search: number of times we sample in each int/float/log list
    """
    configs = []
    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))

    return findBestConfig(train_loader, val_loader, configs, epochs, patience,
                          model_class)


def findBestConfig(train_loader, val_loader, configs, EPOCHS, PATIENCE,
                   model_class):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """

    best_val = None
    best_config = None
    best_model = None
    results = []

    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i + 1), len(configs)), configs[i])

        model = model_class(**configs[i])
        solver = Solver(model, train_loader, val_loader, **configs[i])
        solver.train(epochs=EPOCHS, patience=PATIENCE)
        results.append(solver.best_model_stats)

        if not best_val or solver.best_model_stats["val_loss"] < best_val:
            best_val, best_model, \
            best_config = solver.best_model_stats["val_loss"], model, configs[i]

    print("\nSearch done. Best Val Loss = {}".format(best_val))
    print("Best Config:", best_config)
    return best_model, list(zip(configs, results))


def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """

    config = {}

    for key, (rng, mode) in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <= 0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10 ** (sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config


# def setup_util(n_labeled, batch_size, adam_lr, ema_decay, num_labeled_per_class, alpha, lambda_u, T_Softmax, use_cuda=True):
#     transform_train = transforms.Compose([
#         dataset.RandomPadandCrop(32),
#         dataset.RandomFlip(),
#         dataset.ToTensor(),
#     ])
#
#     transform_val = transforms.Compose([
#         dataset.ToTensor(),
#     ])
#
#     train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10('./data', n_labeled,
#                                                                                     transform_train=transform_train,
#                                                                                     transform_val=transform_val)
#     labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=batch_size, shuffle=True, num_workers=0,
#                                           drop_last=True)
#     unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size, shuffle=True,
#                                             num_workers=0, drop_last=True)
#     val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
#
#     def create_model(ema=False):
#         model = models.WideResNet(num_classes=10)
#         model = model.cuda() if use_cuda else model
#
#         if ema:
#             for param in model.parameters():
#                 param.detach_()
#
#         return model
#
#     model = create_model()
#     ema_model = create_model(ema=True)
#
#     device = 'cuda:0'
#     nb_classes = 10  # number of classes in CIFAR-10 - move it to dataset class!
#     gtg = GTG(nb_classes, max_iter=num_labeled_per_class, device=device).to(device)
#
#     cudnn.benchmark = True
#     print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
#
#     train_criterion = SemiLoss()
#     criterion_gl = nn.NLLLoss().to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=adam_lr)
#
#     ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay)
#     start_epoch = 0
