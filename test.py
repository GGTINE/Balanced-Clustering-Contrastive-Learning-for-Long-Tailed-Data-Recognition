import argparse
from tqdm import tqdm

import torch.optim.optimizer
from torch.cuda.amp import autocast
import torch.nn.functional as F

from main import build_model
from utils import *
from dataset.transform import load_transform
from dataset.loader import load_dataset
from torch.utils.data import DataLoader


def main():
    args = parser.parse_args()

    train_transform, test_transform = load_transform(args)
    train_dataset, test_dataset, num_classes = load_dataset(
        args, train_transform, test_transform
    )

    model = build_model(args, num_classes=num_classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    cls_num_list = train_dataset.cls_num_list
    args.cls_num = len(cls_num_list)

    path = os.path.join(args.path)
    print("=> loading best model '{}'".format(path))
    checkpoint = torch.load(path, map_location="cuda:0")
    model.load_state_dict(checkpoint["state_dict"])

    # evaluate on validation set
    acc1, many, med, few = evaluate(
        train_loader,
        test_loader,
        model,
        args,
    )

    print(
        "Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}".format(
            acc1, many, med, few
        )
    )
# ============================================================================= #


def evaluate(
    train_loader,
    test_loader,
    model,
    args,
):
    model.eval()
    top1 = AverageMeter("Acc@1", ":6.2f")
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    pbar = tqdm(
        test_loader,
        desc=f"Evaluation...",
        ncols=150,
    )

    with torch.no_grad():
        for i, data in enumerate(pbar):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)

            with autocast():
                _, logits = model(inputs)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

            pbar.set_description(
                f"Prec@1({top1.avg:.3f}, {top1.val:.3f})"
            )

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(
            preds, total_labels, train_loader, acc_per_cls=False
        )
        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100", "inat", "imagenet"],
    )
    parser.add_argument("--data", default="data", metavar="DIR")
    parser.add_argument(
        "--arch",
        default="resnet32",
        choices=["resnet32", "resnet50", "resnext50"],
    )
    parser.add_argument(
        "--imb_factor",
        type=float,
        default=0.1,
        help="dataset imbalance factor(0.01 == 100, 0.02 == 50, 0.1 == 10)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--path', default='model path', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument("--randaug_m", default=10, type=int, help="randaug-m")
    parser.add_argument("--randaug_n", default=2, type=int, help="randaug-n")
    parser.add_argument(
        "--cl_views",
        default="sim-sim",
        type=str,
        choices=["sim-sim", "sim-rand", "rand-rand"],
        help="Augmentation strategy for contrastive learning views",
    )
    parser.add_argument(
        "--feat_dim", default=128, type=int, help="feature dimension of mlp head"
    )
    parser.add_argument(
        "--use_norm", default=True, type=bool, help="cosine classifier."
    )

    main()
