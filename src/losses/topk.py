"""
Losses that only take the k worst error into account
Idea coming from https://arxiv.org/abs/2103.14127
"""
import torch
import torch.nn as nn

from losses.x_entropy import FocalLoss



class _Topk(nn.modules.loss._Loss):
    def __init__(self, k=128, sorted=False, reduction="mean"):
        super(_Topk, self).__init__()
        self.k = k
        self.sorted = sorted
        self.reduction = reduction

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        #FIXME? Should we keep the batch shape
        loss = torch.topk(loss.flatten(), self.k, sorted=self.sorted).values
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class _TopkWeighted(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, k=128, sorted=False, reduction="mean"):
        weight = torch.Tensor(weight) if weight is not None else torch.Tensor([1, 1])
        super(_TopkWeighted, self).__init__(weight)
        self.k = k
        self.sorted = sorted
        self.reduction = reduction

    def forward(self, pred, target):
        l1 = self.loss1(pred, target)
        l2 = self.loss2(pred, target)
        loss = self.weight[0] * l1 + self.weight[1] * l2
        loss = torch.topk(loss.flatten(), self.k, sorted=self.sorted).values
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TopkFocalLoss(_Topk):
    """ Only use k worst values to compute FocalLoss """
    def __init__(self, k=128, sorted=False, reduction="mean",
                 # FocalLoss parameters
                 alpha=0.25, gamma=2, ignore_index=-100, label_smoothing=0):
        super(TopkFocalLoss, self).__init__(k, sorted, reduction)
        self.loss = FocalLoss(alpha, gamma, size_average=None, ignore_index=-100,
                              reduce=None, reduction="none", label_smoothing=0)
