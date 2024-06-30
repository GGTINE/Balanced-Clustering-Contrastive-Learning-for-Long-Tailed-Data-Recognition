import argparse


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100", "inat", "imagenet"],
    )
    parser.add_argument("--data", default="data", metavar="DIR")
    parser.add_argument(
        "--arch",
        default="resnet32",
        choices=["resnet18", "resnet32", "resnet34", "resnet50", "resnext50"],
    )
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument(
        "--temp",
        default=0.1,
        type=float,
        help="scalar temperature for contrastive learning",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
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
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.15,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--schedule",
        default=[160, 180],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum of SGD solver",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help="supervised contrastive loss weight(mu)",
    )
    parser.add_argument(
        "--beta", default=1.0, type=float, help="cross entropy loss weight(lambda)"
    )
    parser.add_argument("--ceta", default=0.1, type=float, help="average loss weight")
    parser.add_argument(
        "--randaug",
        default=True,
        type=bool,
        help="use RandAugmentation for classification branch",
    )
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
    parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
    parser.add_argument("--root_log", type=str, default="log")
    parser.add_argument(
        "--cos", default=True, type=bool, help="lr decays by cosine scheduler. "
    )
    parser.add_argument(
        "--use_norm", default=True, type=bool, help="cosine classifier."
    )
    parser.add_argument("--randaug_m", default=10, type=int, help="randaug-m")
    parser.add_argument("--randaug_n", default=20, type=int, help="randaug-n")
    parser.add_argument(
        "--seed", default=42, type=int, help="seed for initializing training"
    )
    parser.add_argument(
        "--tb_save",
        action="store_true",
        help="Save tensorboard",
    )
    parser.add_argument(
        "--imb_factor",
        type=float,
        default=0.01,
        help="dataset imbalance factor(0.01 == 100, 0.02 == 50, 0.1 == 10)",
    )
    parser.add_argument(
        "--queue_size", type=int, default=100, help="queue size for clustering average"
    )
    parser.add_argument(
        "--centroid", type=int, default=5, help="number of clustering centroid"
    )

    return parser.parse_args()
