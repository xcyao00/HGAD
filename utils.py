import math
import random
import torch
import numpy as np


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def onehot(l, n_classes, label_smooth=0.02):
    y = torch.cuda.FloatTensor(l.shape[0], n_classes).zero_()
    y.scatter_(1, l.view(-1, 1), 1.)
    if label_smooth:
        y = y * (1 - label_smooth) + label_smooth / n_classes
    return y


def setting_lr_parameters(args):
    args.scaled_lr_decay_epochs = [i*args.meta_epochs // 100 for i in args.lr_decay_epochs]
    print('LR schedule: {}'.format(args.scaled_lr_decay_epochs))
    if args.lr_warm:
        args.lr_warmup_from = args.lr / 10.0
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warm_epochs / args.meta_epochs)) / 2
        else:
            args.lr_warmup_to = args.lr


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.scaled_lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate