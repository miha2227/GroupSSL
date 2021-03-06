from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wideresnet as models
import dataset.cifar10 as dataset
from progress.bar import Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

from group_loss.gtg import GTG
from utils.misc import get_labeled_and_unlabeled_points


def args_setup():
    # region: arguments
    parser = argparse.ArgumentParser(description='PyTorch MixMatch with Group Loss Training')
    # Optimization options
    parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                        metavar='LR', help='initial learning rate')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Miscs
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    # Device options
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # Method options
    parser.add_argument('--n-labeled', type=int, default=250,
                        help='Number of labeled data')
    parser.add_argument('--train-iteration', type=int, default=1024,
                        help='Number of iteration per epoch')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)

    # Group Loss options
    parser.add_argument('--num-labeled-per-class', type=int, default=2,
                        help='Number of labeled samples per class for group loss')
    parser.add_argument('--T-softmax', type=float, default=10,
                        help='Softmax temperature for group loss')
    # endregion

    # TODO: 1. implement random search for next hyper parameters:
    #  - T-softmax
    #  - num-labeled-per-class (but here be careful because the more correct labels are provided the less info is left to learn
    #    by Group Loss
    #  - alpha (read MixMatch paper)
    #  - lambda-u (read MixMatch paper)
    #  - ema-decay (optional for tuning)

    # TODO: 2. It might happen that in Colab notebook the model learning can stop, due to overflow in the output.
    # That happened to me after 60 epochs of model training. If this is a case remove Bar outputs and print only
    # essential info (e.g. epoch, train, valid and test losses/accuracies on a single output line).

    # region: setup
    args = parser.parse_args()
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    return args


#args = args_setup()
#state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#use_cuda = torch.cuda.is_available()


# endregion
def create_model(use_cuda, ema=False):
    model = models.WideResNet(num_classes=10)
    model = model.cuda() if use_cuda else model

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def main(args, use_cuda):
    state = {k: v for k, v in args._get_kwargs()}

    # global best_acc
    best_acc = 0  # best validation accuracy
    best_t_acc = 0  # best test accuracy
    # Random seed
    np.random.seed(args.manualSeed)
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
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
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("==> creating WRN-28-2")

    model = create_model(use_cuda)
    ema_model = create_model(use_cuda, ema=True)

    device = 'cuda:0'
    nb_classes = 10  # number of classes in CIFAR-10 - move it to dataset class!
    gtg = GTG(nb_classes, max_iter=args.num_labeled_per_class, device=device).to(device)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    train_criterion = SemiLoss(args=args)
    criterion_gl = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer = WeightEMA(model, ema_model, args, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'noisy-cifar-10'
    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(
            ['Train Loss', 'Train Loss X', 'Train Loss X_nll', 'Train Loss X_ce', 'Train Loss U', 'Valid Loss',
             'Valid Acc.', 'Test Loss', 'Test Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    val_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, args.lr))

        train_loss, train_loss_x, train_loss_nll, train_loss_ce, train_loss_u = train(labeled_trainloader, unlabeled_trainloader,
                                                       model, optimizer, ema_optimizer, train_criterion,
                                                       gtg, criterion_gl, criterion, epoch, use_cuda, args=args)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats',args=args)
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats',args=args)
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats',args=args)

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

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        best_t_acc = max(test_acc, best_t_acc)

        #save_checkpoint({
        #    'epoch': epoch + 1,
        #    'state_dict': model.state_dict(),
        #    'ema_state_dict': ema_model.state_dict(),
        #    'acc': val_acc,
        #    'best_acc': best_acc,
        #    'optimizer': optimizer.state_dict(),
        #}, is_best, args.out)
        test_accs.append(test_acc)
        val_accs.append(val_acc)
    logger.close()
    writer.close()

    print('Best val acc: {}'.format(best_acc))
    mean_val_acc = np.mean(val_accs[-20:])
    #mean_test_acc = np.mean(test_accs[-20:])
    print('Mean val acc: {}'.format(mean_val_acc))
    print('Mean test acc: {}'.format(best_t_acc))
    return best_acc, mean_val_acc, best_t_acc


def train(labeled_trainloader, unlabeled_trainloader, model,
          optimizer, ema_optimizer, criterion,
          gtg, criterion_gl, loss_func,
          epoch, use_cuda, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_x_nll = AverageMeter()
    losses_x_ce = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    with open('{}_log.txt'.format(args.out), 'a') as f:
        f.write('\nEpoch: {}'.format(epoch))

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_int = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_int.view(-1, 1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabeled samples
            outputs_u, _ = model(inputs_u)
            outputs_u2, _ = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()  # check how they look like

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        # apply random permutation (shuffle the samples)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        model_out, model_embed = model(mixed_input[0])
        logits = [model_out]
        for input in mixed_input[1:]:
            model_out, _ = model(input)
            logits.append(model_out)

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]  # logits for labeled samples
        logits_u = torch.cat(logits[1:], dim=0)  # logits for unlabeled samples

        Lx, Lx_nll, Lx_ce, Lu, w = criterion(logits_x,
                              mixed_target[:batch_size],
                              targets_int,
                              model_embed,
                              gtg,
                              criterion_gl,
                              loss_func,
                              logits_u,
                              mixed_target[batch_size:],
                              epoch + batch_idx / args.train_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_x_nll.update(Lx_nll.item(), inputs_x.size(0))
        losses_x_ce.update(Lx_ce.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
            batch=batch_idx + 1,
            size=args.train_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w=ws.avg,
        )
        bar.next()
    bar.finish()

    return losses.avg, losses_x.avg, losses_x_nll.avg, losses_x_ce.avg, losses_u.avg


def validate(valloader, model, criterion, epoch, use_cuda, mode, args=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets.long())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()

        with open('{}_log.txt'.format(args.out), 'a') as f:
            f.write('\n {mode}: Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    mode=mode,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def __call__(self,
                 outputs_x,
                 targets_x,
                 orig_targets_x,
                 model_embeddings,
                 gtg,
                 criterion_gl,
                 loss_func,
                 outputs_u,
                 targets_u,
                 epoch,
                 ):
        """
        SemiLoss class which calculates Group Loss for labeled data
        and L2 loss for unlabeled data

        :param outputs_x:
        :param targets_x: one-hot encoded, to get integer labels use torch.argmax(targets_x, dim=1)
        :param orig_targets_x: as integers, and not one-hot encoded
        :param model_embeddings:
        :param gtg: Group Loss
        :param criterion_gl: Negative Log-Likelihood loss function for Group Loss
        :param loss_func: Cross-Entropy loss
        :param outputs_u:
        :param targets_u:
        :param epoch:
        :return:
        """
        labs, L, U = get_labeled_and_unlabeled_points(orig_targets_x,
                                                      num_points_per_class=self.args.num_labeled_per_class,
                                                      num_classes=10)

        # compute normalized softmax
        probs_for_gtg = F.softmax(outputs_x / self.args.T_softmax, dim=1)

        # do GTG (iterative process)
        probs_for_gtg, W = gtg(model_embeddings, model_embeddings.shape[0], labs, L, U, probs_for_gtg)
        probs_for_gtg = torch.log(probs_for_gtg + 1e-12)
        orig_targets_x = orig_targets_x.cuda()
        Lx_nll = criterion_gl(probs_for_gtg, orig_targets_x.long())
        Lx_ce = loss_func(outputs_x, orig_targets_x.long())
        Lx = Lx_nll + Lx_ce
        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lx_nll, Lx_ce, Lu, self.args.lambda_u * linear_rampup(epoch, self.args.epochs)


class WeightEMA(object):
    def __init__(self, model, ema_model, args, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * 0.002 #args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    args = args_setup()
    main(args, use_cuda=True)
