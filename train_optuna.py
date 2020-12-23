import optuna
import time
import random
from train import args_setup, WeightEMA, train, validate, SemiLoss
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import dataset.cifar10 as dataset
import torch.backends.cudnn as cudnn
import models.wideresnet as models
import numpy as np
from group_loss.gtg import GTG


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
                                                                                    transform_val=transform_val)
    n_classes = len(np.unique(train_labeled_set.targets))  # number of classes in the dataset
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, n_classes


def create_model(use_cuda, n_classes, ema=False):
    model = models.WideResNet(num_classes=n_classes)
    model = model.cuda() if use_cuda else model

    if ema:  # todo: ask doubt? Where is the average taking place?
        for param in model.parameters():
            param.detach_()

    return model


def setup_models(use_cuda, n_classes):
    print("==> creating WRN-28-2")
    model = create_model(use_cuda, n_classes)
    ema_model = create_model(use_cuda, n_classes, ema=True)
    return model, ema_model


def train_GroupSSL(trial):  # Objective wrapper for Optuna to tune.
    # region setup args and environment
    args = args_setup()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = 'cuda:0'
        cudnn.benchmark = True
    else:
        device = 'cpu'
        cudnn.benchmark = False

    # endregion

    # Warning!!!: Careful about the default values of the arguments, specially number_of_epochs and out, while running this
    # override number of epoch, number of labled, out folder
    args.epochs = 25
    args.n_labeled = 250
    #args.out = 'test_optuna_fixed_random_seed'
    args.out = 'FixedSed_Optuna_60_250_with_pruning_andShorter_range_fixedSeed'

    # region override args of interest with hyper params to tune from optuna suggest
    args.T_softmax = trial.suggest_int('T_softmax', 20, 100)
    # args.num_labeled_per_class = trial.suggest_int('num_labeled_per_class', 1, 5)
    args.alpha = trial.suggest_float('alpha', 0.7, 0.9)
    args.lambda_u = trial.suggest_int('lambda_u', 35, 105)
    args.lr = trial.suggest_float('lr', 0.001, 0.05)
    # endregion

    setup_random_seed(args.manualSeed)

    # region setup data
    labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, n_classes = setup_data(args)
    # endregion

    # region setup model(s)
    model, ema_model = setup_models(use_cuda, n_classes)
    # endregion

    # region setup criterion(s) or loss(es)  #TODO: changes might be necessary in ths section
    gtg = GTG(n_classes, max_iter=args.num_labeled_per_class, device=device).to(device)  # Group Loss replicator dynamics (check if it is exactly same as original group loss)
    train_criterion = SemiLoss(args=args)
    criterion_gl = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # endregion

    # region setup optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # shall we also try different optimizer?
    ema_optimizer = WeightEMA(model, ema_model, args, alpha=args.ema_decay)
    # endregion

    # region training loop (iterate over epochs):
    test_accs = []
    val_accs = []
    best_acc = 0
    best_t_acc = 0

    for epoch in range(args.epochs):
        print('\nRunning epoch {} of {}:\n====================='.format(epoch+1, args.epochs))
        # train loss(es)
        train_loss, train_loss_x, train_loss_nll, train_loss_ce, train_loss_u = train(labeled_trainloader,
                                                                                      unlabeled_trainloader,
                                                                                      model, optimizer, ema_optimizer,
                                                                                      train_criterion,
                                                                                      gtg, criterion_gl, criterion,
                                                                                      epoch, use_cuda, args=args,
                                                                                      log_enabled=False)

        # train accuracy
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats', args=args, log_enabled=False)

        # validation loss and accuracy
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats', args=args, log_enabled=False)

        # test loss and accuracy (TODO: ask doubt -> why do we need it? what are we doing with it?)
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats', args=args, log_enabled=False)

        # store losses, acc. etc
        best_acc = max(val_acc, best_acc)
        best_t_acc = max(test_acc, best_t_acc)
        test_accs.append(test_acc)
        val_accs.append(val_acc)

        # report for pruning
        trial.report(best_acc, epoch)  # need to think more

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    # endregion

    # region post_processing
    # calculate best, average of loss and accuracies
    mean_val_acc = np.mean(val_accs[-20:])  # todo:ask doubt-> Do we also need to use best validation acc?
    mean_test_acc = np.mean(test_accs[-20:])
    evaluation_score = best_acc  # TODO: set this to the judging performance measure, best validation accuracy wil be used later
    unset_random_seed()  # not to influence the processes of choosing next configuration of hyper parameters
    # endregion

    # return score of the objective function, in this particular case mean validation accuracy
    return evaluation_score


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', study_name='tune_GroupSSL_60_250_with_pruning_fixed_seed',
                                #storage='sqlite:///GroupSSL_optuna_with_pruning_and_fixed_seed.db', load_if_exists=True)
                                storage='sqlite:///GroupSSL_HypParLog_60_250_with_pruning_fixed_seed.db', load_if_exists=True)
    study.optimize(train_GroupSSL, n_trials=30)  # if runningg in P paralel proceses, the set n_trials = (intended number of trials) / P
    time.sleep(1)
    print('\nBest objective value: {}'.format(study.best_value))
    print('\nBest trial: {}'.format(str(study.best_trial)))
    print('\n\nBest parameter value: {}'.format(str(study.best_params)))
