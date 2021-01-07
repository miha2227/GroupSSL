from utils.data_util import setup_data, get_anchor_and_nonanchor_points
from utils.hyper_param_util import setup_args
from utils.misc import setup_device, setup_random_seed, AverageMeter, mkdir_p
from utils.eval import accuracy
from utils.model_util import create_model
from utils import Logger
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop, Adagrad
from progress.bar import Bar
import time
from group_loss.gtg import GTG
import numpy as np
import torch.backends.cudnn as cudnn
from train import WeightEMA, linear_rampup
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR


def guess_label(augmented_images, model, n_augmentations=2, sharpen_temperature=0.5, repeat_tuples_in_op=False):
    assert len(augmented_images) == n_augmentations, "number of images must be equal to number of augmentations"
    with torch.no_grad():
        p = 0
        for img in augmented_images:
            output, _ = model(img)
            p = p + torch.softmax(output, dim=1)
        p = p / n_augmentations
        pt = p ** (1 / sharpen_temperature)
        targets_u = pt / pt.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()
        if repeat_tuples_in_op:
            return (
                       targets_u,) * n_augmentations  # repeating the targets as all augmented versions should have the same targets
        return targets_u


def calculate_supervised_loss(ground_truth, probs, probs_for_gtg, criterion1=None,
                              criterion2=None, scaling_factor=1.0, use_cuda=False):
    if criterion1 is None:
        criterion1 = nn.NLLLoss()
    if criterion2 is None:
        criterion2 = nn.CrossEntropyLoss()
    device = 'cuda:0' if use_cuda else 'cpu'
    criterion1.to(device)
    criterion2.to(device)
    L_s_nll = criterion1(probs_for_gtg, ground_truth)
    L_s_ce = criterion2(probs, ground_truth)
    L_s = scaling_factor * L_s_nll + L_s_ce
    return L_s, L_s_nll, L_s_ce


def calculate_unsupervised_loss(ground_truth, predictions,
                                use_cuda=False):  # both ground truth and predictions should be of same size and should be on same device as 'device'
    device = 'cuda:0' if use_cuda else 'cpu'
    kld = nn.KLDivLoss(reduction='batchmean').to(device)

    jsd = (0.5 * kld(F.log_softmax(predictions, dim=1), ground_truth)) + (
                0.5 * kld(F.log_softmax(ground_truth, dim=1), predictions))

    return jsd


def calculate_unsupervised_loss2(ground_truth, predictions,
                                 use_cuda=False):  # both ground truth and predictions should be of same size and should be on same device as 'device'
    device = 'cuda:0' if use_cuda else 'cpu'
    loss_func = nn.MSELoss().to(device)

    # jsd = (0.5 * kld(F.log_softmax(predictions, dim=1), ground_truth)) + (0.5 * kld(F.log_softmax(ground_truth, dim=1), predictions))

    return loss_func(predictions, ground_truth)


def train(labeled_trainloader, unlabeled_trainloader, model,
          optimizer, ema_optimizer, gtg,
          epoch, use_cuda, args, n_classes=10, lr_scheduler=None, log_enabled=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    L_ss = AverageMeter()
    L_ss_nll = AverageMeter()
    L_ss_ce = AverageMeter()
    L_us = AverageMeter()
    losses = AverageMeter()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    end = time.time()

    bar = Bar('Training', max=args.train_iteration)

    # log epochs and learning rate
    if log_enabled:
        with open(os.path.join(args.out, 'experiment_detail_log.txt'), 'a') as f:
            f.write('\nEpoch: {}'.format(epoch))
            if lr_scheduler:
                f.write('\nCurrent LR: {}'.format(optimizer.param_groups[0]['lr']))

    model.train()  # putting the model in training mode. Actual training is not happening here ;)

    # torch.autograd.set_detect_anomaly(True)  # debug

    for batch_idx in range(args.train_iteration):

        optimizer.zero_grad()

        try:
            inputs_x, targets_int = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_int = labeled_train_iter.next()

        try:
            unlabeled_inputs, _ = unlabeled_train_iter.next()  # unlabeled_inputs is a tuple augmented versions
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            unlabeled_inputs, _ = unlabeled_train_iter.next()  # unlabeled_inputs is a tuple augmented versions

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_int.view(-1, 1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            unlabeled_inputs = tuple(inputs_ui.cuda() for inputs_ui in unlabeled_inputs)
            targets_int = targets_int.cuda()

        guessed_lables = guess_label(unlabeled_inputs, model, args.K, args.T, repeat_tuples_in_op=True)

        # inputs_u = torch.cat([*unlabeled_inputs], dim=0)
        targets_u = torch.cat([*guessed_lables], dim=0)
        all_inputs = torch.cat([inputs_x, *unlabeled_inputs], dim=0)
        probs, embeddings = model(all_inputs)
        # compute normalized softmax
        probs_for_gtg = F.softmax(probs / args.T_softmax, dim=1)

        anchor_lables, anchor_points, non_anchor_points = get_anchor_and_nonanchor_points(targets_int,
                                                                                          args.num_labeled_per_class,
                                                                                          n_classes)
        non_anchor_points.extend(  # concatenating the indices of unlabeled augmented samples
            range(
                targets_int.shape[0],
                targets_u.shape[0] + targets_int.shape[0]
            ))

        # do GTG (iterative process)
        probs_for_gtg, W = gtg(embeddings, embeddings.shape[0], anchor_lables, anchor_points, non_anchor_points,
                               probs_for_gtg)

        probs_for_gtg_unsupervised = probs_for_gtg[targets_int.shape[0]:, :]

        probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

        # supervised loss:
        L_s, L_s_nll, L_s_ce = calculate_supervised_loss(targets_int.long(), probs[:targets_int.shape[0], :],
                                                         probs_for_gtg[:targets_int.shape[0], :], use_cuda=use_cuda)

        # unsupervised loss:
        L_u = calculate_unsupervised_loss(targets_u, probs_for_gtg_unsupervised, use_cuda)

        # Combined Loss:
        # loss = L_s + args.lambda_u * L_u  # todo: also experiment with unit weight
        loss = L_s + args.lambda_u * linear_rampup(epoch, args.epochs) * L_u

        # store losses
        L_ss.update(L_s.item(), batch_size)
        L_ss_nll.update(L_s_nll.item(), batch_size)
        L_ss_ce.update(L_s_ce.item(), batch_size)
        L_us.update(L_u.item(), batch_size)
        losses.update(loss, batch_size)

        # compute gradient and update weights
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_s: {loss_s:.4f} | Loss_s_nll: {loss_s_nll:.4f} | Loss_s_ce: {loss_s_ce:.4f} | Loss_u: {loss_u:.4f}'.format(
            batch=batch_idx + 1,
            size=args.train_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_s=L_ss.avg,
            loss_s_nll=L_ss_nll.avg,
            loss_s_ce=L_ss_ce.avg,
            loss_u=L_us.avg
        )
        bar.next()
        if lr_scheduler:  # learning rate scheduling
            lr_scheduler.step()
    bar.finish()

    return losses.avg, L_ss.avg, L_ss_nll.avg, L_ss_ce.avg, L_us.avg


def validate(valloader, model, criterion, epoch, use_cuda, mode, args=None, log_enabled=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_is_training = model.training
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

        if log_enabled:
            with open(os.path.join(args.out, 'experiment_detail_log.txt'), 'a') as f:
                f.write('\n {mode}: Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    mode=mode,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        model.train(model_is_training)

    return losses.avg, top1.avg


def main(logEnabled=True, args=None, isOptuna = False, no_jsd=False):
    timestamp = datetime.now()
    if args is None:  # args is not none when called from optuna, as hyper params come from optuna suggest
        args = setup_args()
    device, use_cuda = setup_device()
    setup_random_seed(args.manualSeed)
    labeled_trainloader, unlabeled_trainloader, val_loader, test_loader, n_classes = setup_data(args)
    model = create_model(use_cuda, n_classes)
    ema_model = create_model(use_cuda, n_classes, ema=True)
    gtg = GTG(n_classes, max_iter=args.max_iter_gtg, device=device).to(  # max_iter=args.num_labeled_per_class # todo: ask why
        device)
    optimizer = Adam(model.parameters(), lr=args.lr)  # todo: check if we need weight decay
    ema_optimizer = WeightEMA(model, ema_model, args, alpha=args.ema_decay)

    criterion = nn.CrossEntropyLoss().to(device)
    test_accs = []
    val_accs = []
    best_acc = 0
    best_t_acc = 0
    cudnn.benchmark = True

    # control if unsupervised loss is taken into consideration
    if no_jsd:
        args.lambda_u = 0.0

    # region logging setup
    if logEnabled:
        if not os.path.isdir(args.out):
            mkdir_p(args.out)
        title = 'gl_jsd_without_mixup_cifar10'
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(
            ['Train Loss', 'Train Loss S', 'Train Loss S_nll', 'Train Loss S_ce', 'Train Loss U', 'Valid Loss',
             'Valid Acc.', 'Test Loss', 'Test Acc.'])
        writer = SummaryWriter(args.out)  # tensorboard writer
        with open(os.path.join(args.out, 'experiment_detail_log.txt'), 'a') as f:
            f.write('Execution triggered at: {}\n'.format(timestamp.strftime("%Y-%m-%d,%H:%M:%S")))
            f.write('\nExperiment Configuration:\n========================\n')
            f.write(str(args))
            f.write('\n\nEpochs:\n==============================\n')

    # endregion

    for epoch in range(args.epochs):
        print('\nRunning epoch {} of {}:\n====================='.format(epoch + 1, args.epochs))
        # train loss(es)
        train_loss, train_loss_s, train_loss_s_nll, train_loss_s_ce, train_loss_u = train(labeled_trainloader,
                                                                                          unlabeled_trainloader,
                                                                                          model, optimizer,
                                                                                          ema_optimizer,
                                                                                          gtg,
                                                                                          epoch, use_cuda, args=args,
                                                                                          log_enabled=logEnabled)

        # train accuracy
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats',
                                args=args, log_enabled=logEnabled)

        # validation loss and accuracy
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats', args=args,
                                     log_enabled=logEnabled)

        # test loss and accuracy (TODO: ask doubt -> why do we need it? what are we doing with it?)
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats', args=args,
                                       log_enabled=logEnabled)

        # region record logs
        if logEnabled:
            step = args.train_iteration * (epoch + 1)

            writer.add_scalar('losses/train_loss', train_loss, step)
            writer.add_scalar('losses/train_loss_s', train_loss_s, step)
            writer.add_scalar('losses/train_loss_s_nll', train_loss_s_nll, step)
            writer.add_scalar('losses/train_loss_s_ce', train_loss_s_ce, step)
            writer.add_scalar('losses/train_loss_u', train_loss_u, step)
            writer.add_scalar('losses/valid_loss', val_loss, step)
            writer.add_scalar('losses/test_loss', test_loss, step)

            writer.add_scalar('accuracy/train_acc', train_acc, step)
            writer.add_scalar('accuracy/val_acc', val_acc, step)
            writer.add_scalar('accuracy/test_acc', test_acc, step)

            # append logger file
            logger.append([train_loss, train_loss_s, train_loss_s_nll, train_loss_s_ce, train_loss_u, val_loss, val_acc,
                           test_loss, test_acc])
        # endregion

        # store losses, acc. etc
        best_acc = max(val_acc, best_acc)
        best_t_acc = max(test_acc, best_t_acc)
        test_accs.append(test_acc)
        val_accs.append(val_acc)

        #if isOptuna:
        #    yield best_acc, np.mean(val_accs[-20:]), best_t_acc

    # print summary
    print('Best val acc: {}'.format(best_acc))
    mean_val_acc = np.mean(val_accs[-20:])
    # mean_test_acc = np.mean(test_accs[-20:])
    print('Mean val acc: {}'.format(mean_val_acc))
    print('Best test acc: {}'.format(best_t_acc))

    # write overall results and close loggers
    if logEnabled:
        with open(os.path.join(args.out, 'experiment_detail_log.txt'), 'a') as f:  #todo: try to use some kind of global macro definition for this line
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
    return best_acc, mean_val_acc, best_t_acc

if __name__ == '__main__':
    main()  #todo: do we need to implement interleave for batch-norm
    # main(no_jsd=True)
