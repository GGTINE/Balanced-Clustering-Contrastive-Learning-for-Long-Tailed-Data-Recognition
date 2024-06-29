from torchvision.transforms import transforms
from randaugment import rand_augment_transform, Cutout, CIFAR10Policy


def load_transform(args):
    if "cifar" in args.dataset:
        # if args.dataset == "cifar10":
        #     mean = (0.49139968, 0.48215827, 0.44653124)
        #     std = (0.24703233, 0.24348505, 0.26158768)
        #
        # else:  # cifar100
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        normalize = transforms.Normalize(mean=mean, std=std)
        #random_crop = transforms.RandomCrop(32, padding=4)
        random_crop = transforms.RandomResizedCrop(size=32)
        ra_params = dict(
            translate_const=int(32 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    else:
        if args.dataset == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        else:  # inat
            mean = (0.466, 0.471, 0.380)
            std = (0.195, 0.194, 0.192)

        normalize = transforms.Normalize(mean=mean, std=std)
        random_crop = transforms.RandomResizedCrop(224, scale=(0.08, 1.0))
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    augmentation_cifar1 = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize,
    ]
    augmentation_randncls = [
        random_crop,
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
        rand_augment_transform(
            "rand-n{}-m{}-mstd0.5".format(args.randaug_n, args.randaug_m), ra_params
        ),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_randnclsstack = [
        random_crop,
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform(
            "rand-n{}-m{}-mstd0.5".format(args.randaug_n, args.randaug_m), ra_params
        ),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        random_crop,
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ]

    if args.cl_views == "sim-sim":
        if "cifar" in args.dataset:
            transform_train = [
                transforms.Compose(augmentation_cifar1),
                transforms.Compose(augmentation_sim)
            ]
        else:
            transform_train = [
                transforms.Compose(augmentation_sim),
                transforms.Compose(augmentation_sim),
            ]
    elif args.cl_views == "sim-rand" and args.randaug:
        transform_train = [
            transforms.Compose(augmentation_randncls),
            # transforms.Compose(augmentation_randnclsstack),
            transforms.Compose(augmentation_sim),
        ]
    elif args.cl_views == "randstack-randstack" and args.randaug:
        transform_train = [
            # transforms.Compose(augmentation_randncls),
            transforms.Compose(augmentation_randnclsstack),
            transforms.Compose(augmentation_randnclsstack),
        ]
    else:
        raise NotImplementedError(
            "This augmentations strategy is not available for contrastive learning branch!"
        )

    return transform_train, test_transform
