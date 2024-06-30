import torch.nn as nn
import torch.nn.functional as F

from model.resnet import NormedLinear, resnet18, resnet34, resnet50, resnext50
from model.resnet_cifar import resnet32


model_dict = {
    "resnet18": [resnet18, 512],
    "resnet32": [resnet32, 64],
    "resnet34": [resnet34, 512],
    "resnet50": [resnet50, 2048],
    "resnext50": [resnext50, 2048],
}


class Model(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        name="resnet50",
        head="mlp",
        use_norm=True,
        feat_dim=1024,
    ):
        super(Model, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.use_norm = use_norm
        if head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
            )
        elif head == "cifar":
            dim_hidden = 512
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_hidden),
                nn.BatchNorm1d(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden, feat_dim),
            )
        else:
            raise NotImplementedError("head not supported")
        if use_norm:
            self.fc = NormedLinear(dim_in, num_classes)
        else:
            self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat)

        return feat_mlp, logits
