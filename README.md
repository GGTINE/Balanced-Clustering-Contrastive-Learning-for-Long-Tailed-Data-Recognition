# Balanced-Clustering-Contrastive-Learning-for-Long-Tailed-Data-Recognition

## Abstract
Real-world deep learning training data often follow a long-tailed (LT) distribution, where a few classes (head classes) have the most samples and many classes (tail classes) have very few samples. Models trained on LT datasets typically achieve high accuracy on head classes, but suffer from poor performance on tail classes. To address this challenge, strategies based on supervised contrastive learning have been explored. However, existing methods primarily focus on reducing the feature space of head classes and fail to sufficiently expand the feature space of tail classes. In this paper, we propose balanced clustering contrastive learning (BCCL) to balance the feature space between the head and tail classes more effectively. The proposed approach introduces two main components. First, we employ queue-based clustering to extract multiple centroids. This addresses the intra-minibatch class absence issue and maintains intra-class balance. Second, we expand the feature space of tail classes based on class frequency to enhance their expressiveness. An evaluation of four LT datasets, CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and iNaturalist 2018, demonstrates that BCCL consistently outperforms the existing methods. These results establish the ability of BCCL to maintain a balanced feature space in diverse environments.

## Requirement
- pytorch>=1.6.0
- torchvision
- tensorboardX

## Results and Pretrained Models

### Cifar-10-LT
 | Method | imbalance factor |Epochs| Model | Top-1 Acc(%) | link | 
 | :---: | :---: |:---: | :---: | :---: | :---: | 
 |BCCL| 100  | 200 | ResNet-32   | 85.22 | [download](https://drive.google.com/file/d/1-0C62I1OY12hD-ici96I-hfc2k_VHS9J/view?usp=drive_link) | 
 |BCCL| 50 | 200 | ResNet-32   | 88.14 | [download](https://drive.google.com/file/d/1C54hoCFPCok4wOuLddlX9z1rftrFWH4-/view?usp=drive_link)|
 |BCCL| 10 | 200 | ResNet-32   | 91.92 | [download](https://drive.google.com/file/d/1oIqt0l08wbyW88XU-d4U6Fv1JV00OWty/view?usp=drive_link) |

 ### Cifar-100-LT
 | Method | imbalance factor |Epochs| Model | Top-1 Acc(%) | link | 
 | :---: | :---: |:---: | :---: | :---: | :---: | 
 |BCCL| 100  | 200 | ResNet-32   | 52.55 | [download](https://drive.google.com/file/d/1RN1WSWbRFxA_u5kppWvIKvBd5SJJMm3U/view?usp=drive_link) | 
 |BCCL| 50 | 200 | ResNet-32   | 56.89 | [download](https://drive.google.com/file/d/1fnM4AySDP8CQw-zcfMTGnKP-THgv3eRB/view?usp=drive_link)|
 |BCCL| 10 | 200 | ResNet-32   | 65.3 | [download](https://drive.google.com/file/d/1B6f7zRPL8k52BzfbsMUvRxjoB17kJhzv/view?usp=drive_link) |


## Usage
For Cifar-10-LT and Cifar-100-LT training and evaluation. All experiments are conducted on 1 GPU.

### Cifar-10-LT
To do supervised training with BCCL for 200 epochs on Cifar-10-LT with 1 GPU, run
```
python main.py --dataset cifar10 \
  --arch resnet32 --epochs 200 --temp 0.1 \
  --lr 0.15 --use-norm True \
  --wd 5e-4 --cos True \
  -b 128 --feat_dim 128 --tb_save
```

To evaluate the performance on the test set, run
```
python test.py --path $PATH
```

### Cifar-100-LT
To do supervised training with BCCL for 200 epochs on Cifar-100-LT with 1 GPU, run
```
python main.py --dataset cifar100 \
  --arch resnet32 --epochs 200 --temp 0.1 \
  --lr 0.15 --use-norm True \
  --wd 5e-4 --cos True \
  -b 128 --feat_dim 128 --tb_save
```

### ImageNet-LT
To do supervised training with BCCL for 100 epochs on ImageNet-LT with 1 GPU, run
```
python main.py --dataset imagenet \
  --arch resnet50 --epochs 100 --temp 0.07 \
  --lr 0.1 --use-norm True \
  --wd 5e-4 --cos True \
  -b 128 --feat_dim 2048 --tb_save
```


### For ImageNet-LT & iNaturalist
You should download [ImageNet-LT](http://image-net.org/download) dataset manually, place them in your `data_root`

Long-tailed version will be created using train/val splits (.txt files)

ImageNet-LT download [ImageNet-LT](https://drive.google.com/file/d/1wRxlzWtgyYDIL1Az6dzNqUNGbmT1SKVA/view?usp=drive_link) 

iNaturalist 2018 download [iNaturalist 2018](https://drive.google.com/file/d/1j5T4RFd03-2vjbVta5EpCIKZZsW6K_hn/view?usp=drive_link) 
