import os
import math
import json
import random
import shutil
from datetime import datetime
from PIL import ImageFilter

import numpy as np
import torch


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def shot_acc(
    preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False
):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError("Type ({}) of preds not supported".format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.store_name, "cl_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, filename.replace("cl_ckpt.pth.tar", "best.pth.tar"))


class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform2(x)]


def save_args_to_file(args):
    timestamp = str(datetime.now().timestamp())
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        args.store_name = os.path.join(
            args.root_log, args.dataset, str(args.imb_factor), args.store_name
        )
    else:
        args.store_name = os.path.join(args.root_log, args.dataset, args.store_name)
    os.makedirs(args.store_name, exist_ok=True)

    args.store_name = os.path.join(args.store_name, timestamp)
    file_path = os.path.join(args.store_name, "parser.json")

    args_dict = vars(args)
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        folder_path = "/".join(file_path.split("/")[0:5])
    else:
        folder_path = "/".join(file_path.split("/")[0:4])
    os.makedirs(folder_path, exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(args_dict, file, indent=4)


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs + 1)
                / (args.epochs - args.warmup_epochs + 1)
            )
        )
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def group_features_by_class(features, labels, num_classes):
    labels = labels.repeat(2)

    unique_labels = torch.unique(labels)
    grouped_features = list(range(num_classes))
    for label in unique_labels:
        mask = (labels == label).nonzero().flatten()
        grouped_features[label] = features[mask]

    return grouped_features
