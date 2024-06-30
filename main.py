import time
import warnings
from tqdm import tqdm

from kmeans_gpu import KMeans
import torch.optim.optimizer

from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model.models import Model

from myparser import load_args
from tensorboardX import SummaryWriter

from utils import *
from dataset.transform import load_transform
from dataset.loader import load_dataset
from torch.utils.data import DataLoader

from loss.contrastive import SupConLoss
from loss.logitadjust import LogitAdjust

cudnn.benchmark = True


def main():
    args = load_args()
    args.store_name = "_".join(
        [
            args.arch,
            f"B{str(args.batch_size)}",
            f"lr{str(args.lr)}",
        ]
    )
    save_args_to_file(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main_worker(args)


def main_worker(args):
    # load_dataset
    train_transform, test_transform = load_transform(args)
    train_dataset, test_dataset, num_classes = load_dataset(
        args, train_transform, test_transform
    )

    # build_model
    model = build_model(args, num_classes=num_classes)

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cuda:0")
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    cls_num_list = train_dataset.cls_num_list
    args.cls_num = len(cls_num_list)
    feature_list = None

    # build data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # build loss func
    criterion_ce = LogitAdjust(cls_num_list).cuda()
    criterion_scl = SupConLoss(cls_num_list, args.temp, args.centroid).cuda()
    kmeans = KMeans(n_clusters=args.centroid)

    tf_writer = SummaryWriter(log_dir=args.store_name)

    best_acc1 = 0.0
    best_many, best_med, best_few = 0.0, 0.0, 0.0

    num_per_classes = torch.ones(num_classes)
    feature_average = torch.zeros(
        (num_classes, args.centroid, args.feat_dim)
    )

    # training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)

        # train for one epoch
        num_per_classes, feature_list, feature_average = train(
            train_loader,
            model,
            criterion_ce,
            criterion_scl,
            optimizer,
            epoch,
            args,
            feature_list,
            feature_average,
            kmeans,
            num_classes,
            num_per_classes,
            tf_writer,
        )

        # evaluate on validation set
        acc1, many, med, few = evaluate(
            train_loader,
            test_loader,
            model,
            criterion_ce,
            epoch,
            args,
            tf_writer,
        )

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
        print(
            "Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}".format(
                best_acc1, best_many, best_med, best_few
            )
        )
        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )


def build_model(args, num_classes):
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == "resnet18":
        model = Model(
            name="resnet18",
            num_classes=num_classes,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
        )
    elif args.arch == "resnet32":
        model = Model(
            name="resnet32",
            num_classes=num_classes,
            feat_dim=args.feat_dim,
            head="cifar",
            use_norm=args.use_norm,
        )
    elif args.arch == "resnet34":
        model = Model(
            name="resnet34",
            num_classes=num_classes,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
        )
    elif args.arch == "resnet50":
        model = Model(
            name="resnet50",
            num_classes=num_classes,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
        )
    elif args.arch == "resnext50":
        model = Model(
            name="resnext50",
            num_classes=num_classes,
            feat_dim=args.feat_dim,
            use_norm=args.use_norm,
        )
    else:
        raise NotImplementedError("This model is not supported")
    print(model)
    model = model.cuda()

    return model


def train(
    train_loader,
    model,
    criterion_ce,
    criterion_scl,
    optimizer,
    epoch,
    args,
    feature_list,
    feature_average,
    kmeans,
    num_classes,
    num_per_classes,
    tf_writer,
):
    batch_time = AverageMeter("Time", ":6.3f")
    ce_loss_all = AverageMeter("CE_Loss", ":.4e")
    scl_loss_all = AverageMeter("SCL_Loss", ":.4e")
    centroid_loss_all = AverageMeter("CEN_Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    model.train()
    end = time.time()

    count_labels = torch.zeros(num_classes)

    pbar = tqdm(
        train_loader,
        desc=f"Training(Epoch = {epoch + 1})...",
        ncols=200,
    )
    for i, data in enumerate(pbar):
        inputs, targets = data
        inputs = torch.cat([inputs[0], inputs[1]], dim=0)
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = targets.shape[0]

        with autocast(enabled=True):
            feat_mlp, logits = model(inputs)
            logits, _ = torch.split(logits, [batch_size, batch_size], dim=0)
            f1, f2 = torch.split(feat_mlp, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            ce_loss = criterion_ce(
                logits, targets
            )
            # compute loss
            if args.warmup_epochs > epoch:
                scl_loss = criterion_scl(
                    features,
                    targets,
                    warmup=True,
                )
                loss = args.alpha * scl_loss + args.beta * ce_loss
            else:
                if epoch % 10 == 0:
                    for k in range(num_classes):
                        centroids = kmeans(torch.unsqueeze(feature_list[k], 0))
                        feature_average[k] = torch.squeeze(centroids)
                scl_loss, centroid_loss = criterion_scl(
                    features, targets, average=feature_average.cuda()
                )
                loss = (
                    args.alpha * scl_loss
                    + args.beta * ce_loss
                    + args.ceta * centroid_loss
                )

        with torch.no_grad():
            grouped_features = group_features_by_class(feat_mlp, targets, num_classes)
            if epoch == 0 and i == 0:
                feature_list = grouped_features
            else:
                for j in range(num_classes):
                    if str(type(grouped_features[j])) == "<class 'int'>":
                        continue
                    if (
                        str(type(grouped_features[j])) != "<class 'int'>"
                        and str(type(feature_list[j])) == "<class 'int'>"
                    ):
                        feature_list[j] = grouped_features[j]
                    if (
                        feature_list[j].size(0) + grouped_features[j].size(0)
                        < args.queue_size
                    ):
                        feature_list[j] = torch.cat(
                            [feature_list[j], grouped_features[j]], dim=0
                        )
                    else:
                        if len(grouped_features[j]) >= args.queue_size:
                            feature_list[j] = grouped_features[j][-args.queue_size :]
                        else:
                            feature_list[j] = torch.cat(
                                [
                                    feature_list[j][
                                        -(args.queue_size - len(grouped_features[j])):
                                    ],
                                    grouped_features[j],
                                ],
                                dim=0,
                            )

        # record update
        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)
        if args.warmup_epochs <= epoch:
            centroid_loss_all.update(centroid_loss.item(), batch_size)
        acc1 = accuracy(logits, targets, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.warmup_epochs > epoch:
            pbar.set_description(
                f"Training(E({epoch + 1})): "
                f"BT({batch_time.avg:.3f}, {batch_time.val:.3f}) "
                f"CE({ce_loss_all.avg:.4f}, {ce_loss_all.val:.4f}) "
                f"SCL_v({scl_loss_all.avg:.6f}, {scl_loss_all.val:.6f}) "
                f"Prec@1({top1.avg:.3f}, {top1.val:.3f})"
            )
        else:
            pbar.set_description(
                f"Training(E({epoch + 1})): "
                f"BT({batch_time.avg:.3f}, {batch_time.val:.3f}) "
                f"CE({ce_loss_all.avg:.4f}, {ce_loss_all.val:.4f}) "
                f"SCL_v({scl_loss_all.avg:.6f}, {scl_loss_all.val:.6f}) "
                f"CL({centroid_loss_all.avg:.6f}, {centroid_loss_all.val:.6f} "
                f"Prec@1({top1.avg:.3f}, {top1.val:.3f})"
            )

    num_per_classes = num_per_classes + count_labels.detach().cpu().numpy()

    tf_writer.add_scalar("loss/CE_train", ce_loss_all.avg, epoch)
    tf_writer.add_scalar("loss/SCL_train", scl_loss_all.avg, epoch)
    if args.warmup_epochs <= epoch:
        tf_writer.add_scalar("loss/CL_train", centroid_loss_all.avg, epoch)
    tf_writer.add_scalar("acc/train_top1", top1.avg, epoch)

    return num_per_classes, feature_list, feature_average


def evaluate(
    train_loader,
    test_loader,
    model,
    criterion_ce,
    epoch,
    args,
    tf_writer=None,
):
    model.eval()
    batch_time = AverageMeter("Time", ":6.3f")
    ce_loss_all = AverageMeter("CE_Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    pbar = tqdm(
        test_loader,
        desc=f"Evaluation...",
        ncols=150,
    )

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(pbar):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)

            with autocast():
                _, logits = model(inputs)
                ce_loss = criterion_ce(
                    logits, targets
                )

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            ce_loss_all.update(ce_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            batch_time.update(time.time() - end)

            pbar.set_description(
                f"Evaluating(E({epoch + 1})): "
                f"BT({batch_time.avg:.3f}, {batch_time.val:.3f}) "
                f"CE({ce_loss_all.avg:.4f}, {ce_loss_all.val:.4f}) "
                f"Prec@1({top1.avg:.3f}, {top1.val:.3f})"
            )
        tf_writer.add_scalar("loss/CE_val", ce_loss_all.avg, epoch)
        tf_writer.add_scalar("acc/val_top1", top1.avg, epoch)

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(
            preds, total_labels, train_loader, acc_per_cls=False
        )
        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1


if __name__ == "__main__":
    scaler = GradScaler()
    main()
