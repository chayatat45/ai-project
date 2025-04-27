import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match, encode

class MultiBoxLoss(nn.Module):
    def __init__(self, threshold=0.5, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, predictions, targets):
        loc_preds, conf_preds, priors = predictions
        batch_size = loc_preds.size(0)
        num_priors = priors.size(0)

        loc_targets = torch.zeros_like(loc_preds)
        conf_targets = torch.zeros(batch_size, num_priors, dtype=torch.long).to(conf_preds.device)

        for idx in range(batch_size):
            truths = targets[idx][:, :-1]
            labels = targets[idx][:, -1]
            defaults = priors

            match(self.threshold, truths, defaults, labels, loc_targets, conf_targets, idx)

        pos = conf_targets > 0
        loc_loss = F.smooth_l1_loss(loc_preds[pos], loc_targets[pos], reduction='sum')

        batch_conf = conf_preds.view(-1, conf_preds.size(-1))
        loss_c = F.cross_entropy(batch_conf, conf_targets.view(-1), reduction='none')

        loss_c = loss_c.view(batch_size, -1)
        loss_c[pos] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=pos.size(1) - 1)

        neg = idx_rank < num_neg.expand_as(idx_rank)

        pos_idx = pos.unsqueeze(2).expand_as(conf_preds)
        neg_idx = neg.unsqueeze(2).expand_as(conf_preds)
        conf_loss = F.cross_entropy(conf_preds[(pos_idx + neg_idx).gt(0)].view(-1, conf_preds.size(-1)),
                                    conf_targets[(pos + neg).gt(0)], reduction='sum')

        N = num_pos.data.sum().float()
        loc_loss /= N
        conf_loss /= N

        return loc_loss + conf_loss
