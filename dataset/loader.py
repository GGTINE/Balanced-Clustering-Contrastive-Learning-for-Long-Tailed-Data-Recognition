import os
import random
from PIL import Image

import numpy as np

import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset


def load_dataset(args, train_transform=None, test_transform=None):
    if args.dataset == "cifar10":
        num_classes = 10
        train_dataset = IMBALANCECIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=train_transform,
            imb_factor=args.imb_factor,
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )

    elif args.dataset == "cifar100":
        num_classes = 100
        train_dataset = IMBALANCECIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=train_transform,
            imb_factor=args.imb_factor,
        )
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=test_transform
        )

    elif args.dataset == "imagenet":
        num_classes = 1000
        txt_train = "dataset/ImageNet_LT/ImageNet_LT_train.txt"
        txt_test = "dataset/ImageNet_LT/ImageNet_LT_test.txt"
        train_dataset = ImageNetLT(
            root=args.data + "/image/", txt=txt_train, transform=train_transform
        )
        test_dataset = ImageNetLT(
            root=args.data + "/image/",
            txt=txt_test,
            transform=test_transform,
            train=False,
        )

    elif args.dataset == "inat":
        num_classes = 8142
        txt_train = "dataset/iNaturalist18/iNaturalist18_train.txt"
        txt_test = "dataset/iNaturalist18/iNaturalist18_val.txt"
        train_dataset = INaturalist(
            root=args.data, txt=txt_train, transform=train_transform
        )
        test_dataset = INaturalist(
            root=args.data,
            txt=txt_test,
            transform=test_transform,
            train=False,
        )
    else:
        raise NotImplementedError

    return train_dataset, test_dataset, num_classes


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(
        self,
        root: str,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        imb_type="exp",
        imb_factor=0.01,
        rand_number=0,
    ):
        super().__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)
        self.cls_num_list = self.get_cls_num_list()
        self.labels = self.targets

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * the_img_num
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img, labels = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](img)
                sample2 = self.transform[1](img)
                # sample3 = self.transform[2](img)
                return [sample1, sample2], labels  # , index
            else:
                return self.transform(img), labels


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    cls_num = 100


class INaturalist(Dataset):
    def __init__(self, root, txt, transform=None, train=True):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.num_classes = 8142
        self.train = train
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                return [sample1, sample2], label  # , index
            else:
                return self.transform(sample), label


class ImageNetLT(Dataset):
    def __init__(self, root, txt, transform=None, train=True, class_balance=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.num_classes = 1000
        self.train = train
        self.class_balance = class_balance
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
            label = random.randint(0, self.num_classes - 1)
            index = random.choice(self.class_data[label])
            path = self.img_path[index]

        else:
            path = self.img_path[index]
            label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                return [sample1, sample2], label
            else:
                return self.transform(sample), label
