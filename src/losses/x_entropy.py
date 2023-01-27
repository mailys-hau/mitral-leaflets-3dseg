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


class BalancedEntropyLoss(nn.CrossEntropyLoss):
    """ Computed weights based on the repartition of classes in each sample """
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0):
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
        weight = (1 - target).mean(dim=self.dims).mean(dim=0)
        return F.cross_entropy(pred, target, weight=weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
