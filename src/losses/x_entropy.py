import torch
import torch.nn as nn
import torch.nn.functional as F



class XEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0):
        # Wrapper to accept list as weight parameter
        weight = torch.Tensor(weight) if weight is not None else weight
        super(XEntropyLoss, self).__init__(weight, size_average, ignore_index,
                                           reduce, reduction, label_smoothing)


class FocalLoss(nn.CrossEntropyLoss):
    # Implementation from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    def __init__(self, alpha=0.25, gamma=2, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0):
        super(FocalLoss, self).__init__(None, size_average, ignore_index, reduce,
                                        reduction, label_smoothing)
        #FIXME: Assert values are in proper ranges
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # `torch.nn.CrossEntropy` expect *non normalized* preds
        p = F.softmax(pred, 1)
        # Returns size (B, W, H, D) when inputs are (B, C, W, H, D)
        xentropy = F.cross_entropy(pred, target, weight=None,
                                   ignore_index=self.ignore_index,
                                   label_smoothing=self.label_smoothing, 
                                   reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        # Matching CrossEntropy shape, but does it makes sense?
        p_t, alpha_t = p_t.mean(dim=1), alpha_t.mean(dim=1)
        loss = alpha_t * (xentropy * ((1 - p_t) ** self.gamma))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BalancedEntropyLoss(nn.CrossEntropyLoss):
    """ Computed weights based on the repartition of classes in each sample """
    def __init__(self, size_average=None, ignore_index=-100, reduce=None,
                 reduction="mean", label_smoothing=0):
        # Enforcing there's no weights given, we computed it per data
        super(BalancedEntropyLoss, self).__init__(None, size_average,
                                                  ignore_index, reduce,
                                                  reduction, label_smoothing)

    def forward(self, pred, target):
        # We want the heavier weight on less represented classes
        # First dim is batch size, second is number of classes
        if not hasattr(self, "dims"):
            self.dims = [i for i in range(2, target.ndim)]
        # First average accross the image, then accros the batch
        weight = 1 - target.mean(dim=self.dims).mean(dim=0)
        # Enforce weight.sum() = number of classes
        weight = target.shape[1] * F.softmax(weight, 0)
        return F.cross_entropy(pred, target, weight=weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

class PreservedBalancedEntropyLoss(nn.CrossEntropyLoss):
    """ Similar as BalancedCrossEntropy but aims to keep loss magnitude of
        XEntropyLoss with no weights. """
    def __init__(self, size_average=None, ignore_index=-100, reduce=None,
                 reduction="mean", label_smoothing=0):
        # Enforcing there's no weights given, we computed it per data
        super(PreservedBalancedEntropyLoss, self).__init__(None, size_average,
                                                           ignore_index, reduce,
                                                           reduction, label_smoothing)

    def forward(self, pred, target):
        # We want the heavier weight on less represented classes
        # First dim is batch size, second is number of classes
        if not hasattr(self, "dims"):
            self.dims = [i for i in range(2, target.ndim)]
        # First average accross the image, then accros the batch
        weight = target.mean(dim=self.dims).mean(dim=0)
        # Enforce loss value to be same scale to default CrossEntropyLoss
        weight = 1 / (target.shape[1] * weight)
        return F.cross_entropy(pred, target, weight=weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
