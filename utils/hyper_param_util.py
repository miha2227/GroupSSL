import argparse
import random


def setup_args():
    # region: arguments
    parser = argparse.ArgumentParser(description='PyTorch MixMatch with Group Loss Training')
    # Optimization options
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
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
    parser.add_argument('--out', default='result_gl_jsd_without_mixup',
                        help='Directory to output the result')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--K', type=int, default=2, help='Number of augmentations for unlabeled samples')
    parser.add_argument('--scaling_loss', default=1.0, type=float, dest='scaling_loss',
                        help='Scaling parameter for computing supervised loss')

    # Group Loss options
    parser.add_argument('--num-labeled-per-class', type=int, default=2,
                        help='Number of labeled samples per class for group loss')
    parser.add_argument('--T-softmax', type=float, default=10,
                        help='Softmax temperature for group loss')
    parser.add_argument('--max_iter_gtg', type=int, default=5, help='Number of iterations for gtg replicator dynamics')
    # endregion

    # region: setup
    args = parser.parse_args()
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    return args
