import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1, centroid=10):
        super().__init__()
        self.base_temperature = 0.1
        self.temperature = temperature
        self.cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        self.cos = nn.CosineSimilarity(eps=1e-6).cuda()
        self.centroid = centroid

    def forward(self, features, labels=None, warmup=False, average=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        batch_size = features.shape[0]
        anchor_count = features.shape[1]
        batch_cls_count = (
            torch.eye(len(self.cls_num_list))[labels.cpu()].sum(dim=0).squeeze().cuda()
        )
        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, anchor_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, anchor_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        per_ins_weight = (
            torch.tensor(
                [batch_cls_count[i] for i in labels.repeat(2, 1)], device=device
            )
            .view(1, -1)
            .expand(2 * batch_size, 2 * batch_size)
            - mask
        )
        per_ins_weight = torch.where(per_ins_weight < 1e-6, 1, per_ins_weight)

        exp_logits = torch.exp(logits) * logits_mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        if warmup:
            return loss
        else:
            average_dot = torch.matmul(
                anchor_feature, average.view(-1, average.size(2)).T
            )

            average_logit = average_dot
            average_mask = torch.zeros(
                labels.size(0) * 2, len(self.cls_num_list), device=device
            )
            average_mask.scatter_(1, labels.repeat(2, 1), 1)
            average_mask = torch.repeat_interleave(average_mask, average.size(1)).view(
                average_mask.size(0), -1
            )

            negative_average_mask = 1 - average_mask
            pos_average_logit = torch.exp(average_logit * average_mask)
            neg_average_logit = torch.exp(average_logit * negative_average_mask)

            average_loss = -torch.log(pos_average_logit / neg_average_logit).mean()

            return loss, average_loss
