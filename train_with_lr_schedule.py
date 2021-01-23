from utils.misc import setup_device, mkdir_p
from utils import Logger
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
import os
from group_loss.gtg import GTG
from torch.optim import Adam
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR  #,OneCycleLR
from tensorboardX import SummaryWriter
from utils.pyt import OneCycleLR


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

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def setup_models(use_cuda, n_classes):
    print("==> creating WRN-28-2")
    model = create_model(use_cuda, n_classes)
    ema_model = create_model(use_cuda, n_classes, ema=True)
    return model, ema_model


def train_GroupSSL_with_LR_Schedule(args, logEnabled=True, isOptuna=False):  # Objective wrapper for Optuna to tune.
    timestamp = datetime.now()

    if args is None:  # args is not none when called from optuna, as hyper params come from optuna suggest
        args = args_setup()

    device, use_cuda = setup_device()
    setup_random_seed(args.manualSeed)

    labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, n_classes = setup_data(args)

    model = create_model(use_cuda, n_classes)
    ema_model = create_model(use_cuda, n_classes, ema=True)
    gtg = GTG(n_classes, max_iter=args.max_iter_gtg, device=device).to(device)

    # region define criterion
    train_criterion = SemiLoss(args=args)
    criterion_gl = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # endregion

    optimizer = Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, args, alpha=args.ema_decay)

    # region optim_schedular
    # optim_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=round(args.epochs*0.2), gamma=0.1)  # divides lr by 10 every 20 percent of epochs
    # optim_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=85, gamma=0.5)  # divides lr by 2 after 85 epochs
    # optim_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.7)
    # optim_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)  # divides lr by 10 on plateu
    # optim_schedular = CyclicLR(optimizer, base_lr=1e-3, max_lr=0.09, step_size_up=2048, mode='exp_range', gamma=0.98, cycle_momentum=False)
    # optim_schedular = OneCycleLR(optimizer, max_lr=0.01, epochs=args.epochs, steps_per_epoch=args.train_iteration, cycle_momentum=False)
    # optim_schedular = OneCycleLR(optimizer, max_lr=0.01, epochs=args.epochs, steps_per_epoch=args.train_iteration, final_div_factor=0.188638314328, cycle_momentum=False)
    # optim_schedular = OneCycleLR(optimizer, max_lr=0.01, epochs=args.epochs, steps_per_epoch=args.train_iteration, cycle_momentum=False)
    # optim_schedular = OneCycleLR(optimizer, max_lr=0.01, total_steps=int((args.epochs * args.train_iteration) * 0.75), final_div_factor=0.188638314328, cycle_momentum=False)
    # optim_schedular = CyclicLR(optimizer, base_lr=1e-3, max_lr=0.01, step_size_up=2048, mode='exp_range', gamma=0.98, cycle_momentum=False)  # 20 epochs, activate schedule after 10 epochs
    optim_schedular = CosineAnnealingLR(optimizer, T_max=40960, eta_min=0.00001)
    #optim_schedular = None
    # endregion

    # region logging setup
    if logEnabled:
        if not os.path.isdir(args.out):
            mkdir_p(args.out)
        title = 'GroupSSL_with_Cyclic_LR_Schedule'
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(
            ['Train Loss', 'Train Loss X', 'Train Loss X_nll', 'Train Loss X_ce', 'Train Loss U', 'Valid Loss',
             'Valid Acc.', 'Test Loss', 'Test Acc.'])
        writer = SummaryWriter(args.out)  # tensorboard writer
        with open(os.path.join(args.out, 'experiment_detail_log.txt'), 'a') as f:
            f.write('Execution triggered at: {}\n'.format(timestamp.strftime("%Y-%m-%d,%H:%M:%S")))
            f.write('\nExperiment Configuration:\n========================\n')
            f.write(str(args))
            f.write('\n\nEpochs:\n==============================\n')

    # endregion



    # region training loop (iterate over epochs):
    test_accs = []
    val_accs = []
    best_acc = 0
    best_t_acc = 0

    for epoch in range(args.epochs):
        print('\nRunning epoch {} of {}:\n====================='.format(epoch+1, args.epochs))
        if optim_schedular:
            print('Current LR: {}'.format(optimizer.param_groups[0]['lr']))

        train_loss, train_loss_x, train_loss_nll, train_loss_ce, train_loss_u = train(labeled_trainloader,
                                                                                      unlabeled_trainloader,
                                                                                      model, optimizer, ema_optimizer,
                                                                                      train_criterion,
                                                                                      gtg, criterion_gl, criterion,
                                                                                      epoch, use_cuda, args=args,
                                                                                      lr_scheduler=optim_schedular,
                                                                                      log_enabled=logEnabled,
                                                                                      #isCyclicLRScheduler=True)
                                                                                      isCyclicLRScheduler=False,
                                                                                      ema_model=ema_model)

        # train accuracy
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats', args=args, log_enabled=logEnabled)

        # validation loss and accuracy
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats', args=args, log_enabled=logEnabled)

        # test loss and accuracy (TODO: ask doubt -> why do we need it? what are we doing with it?)
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats', args=args, log_enabled=logEnabled)

        # region record logs
        if logEnabled:
            step = args.train_iteration * (epoch + 1)

            writer.add_scalar('losses/train_loss', train_loss, step)
            writer.add_scalar('losses/train_loss_x', train_loss_x, step)
            writer.add_scalar('losses/train_loss_nll', train_loss_nll, step)
            writer.add_scalar('losses/train_loss_ce', train_loss_ce, step)
            writer.add_scalar('losses/train_loss_u', train_loss_u, step)
            writer.add_scalar('losses/valid_loss', val_loss, step)
            writer.add_scalar('losses/test_loss', test_loss, step)

            writer.add_scalar('accuracy/train_acc', train_acc, step)
            writer.add_scalar('accuracy/val_acc', val_acc, step)
            writer.add_scalar('accuracy/test_acc', test_acc, step)

            # append logger file
            logger.append([train_loss, train_loss_x, train_loss_nll, train_loss_ce, train_loss_u, val_loss, val_acc,
                           test_loss, test_acc])
        # endregion

        # store losses, acc. etc
        best_acc = max(val_acc, best_acc)
        best_t_acc = max(test_acc, best_t_acc)
        test_accs.append(test_acc)
        val_accs.append(val_acc)

        #if isOptuna:
        #    yield best_acc, np.mean(val_accs[-20:]), best_t_acc

        #optim_schedular.step(val_loss)

        #optim_schedular.step()
        if optim_schedular and epoch+1 > 60:
            ema_optimizer.set_LR(optimizer.param_groups[0]['lr'])

    # endregion

    # region post_processing
    # calculate best, average of loss and accuracies
    mean_val_acc = np.mean(val_accs[-20:])
    mean_test_acc = np.mean(test_accs[-20:])
    #evaluation_score = best_acc

    # write overall results and close loggers
    if logEnabled:
        with open(os.path.join(args.out, 'experiment_detail_log.txt'),
                  'a') as f:  # todo: try to use some kind of global macro definition for this line
            f.write('\n\nSummary Results:\n======================')
            f.write(
                '\n Best val acc: {best_acc: .4f} \n Mean val acc: {mean_val_acc: .4f} \n Best test acc: {best_t_acc: .4f}'.format(
                    best_acc=best_acc,
                    mean_val_acc=mean_val_acc,
                    best_t_acc=best_t_acc
                ))
            f.write('\n\nExecution finished at: {}\n'.format(datetime.now().strftime("%Y-%m-%d,%H:%M:%S")))
        logger.close()
        writer.close()
    # endregion

    # return score of the objective function, in this particular case mean validation accuracy
    return best_acc, mean_val_acc, best_t_acc


if __name__ == '__main__':
    args = args_setup()
    best_acc, mean_val_acc, best_t_acc = train_GroupSSL_with_LR_Schedule(args)
    print('\nBest validation accuracy: {}'.format(best_acc))
    print('\nAverage validation accuracy: {}'.format(mean_val_acc))
    print('\n\nBest test accuracy: {}'.format(best_t_acc))
