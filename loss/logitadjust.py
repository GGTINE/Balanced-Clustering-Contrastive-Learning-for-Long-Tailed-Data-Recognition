import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1.0, weight=None):
        super(LogitAdjust, self).__init__()
        self.weight = weight
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)
