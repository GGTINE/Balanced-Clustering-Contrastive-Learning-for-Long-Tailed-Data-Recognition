from tqdm import tqdm
from matplotlib import pyplot as plt

import numpy as np
from sklearn.manifold import TSNE

import torch


def tsne_visualize(model, loader):
    actual = []
    deep_features = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()

            # feature 확인 필요 (2024-01-08), Clustering, RF, compression 등 혹은 logits으로
            # No Meta Validation Set, No memory, Only feature and logits
            # EB 사용
            feat_mlp, logits = model(inputs)

            deep_features += feat_mlp.cpu().numpy().tolist()
            actual += targets.cpu().numpy().tolist()

    tsne = TSNE(n_components=2, random_state=0)
    cluster = np.array(tsne.fit_transform(np.array(deep_features)))
    actual = np.array(actual)

    plt.figure(figsize=(10, 10))
    cifar = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    for i, label in zip(range(10), cifar):
        idx = np.where(actual == i)
        plt.scatter(cluster[idx, 0], cluster[idx, 1], marker=".", label=label)

    plt.legend(loc="upper right")
    plt.savefig("save50.png")
    plt.show()
