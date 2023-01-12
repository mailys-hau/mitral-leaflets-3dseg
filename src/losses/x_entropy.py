import torch
import torch.nn as nn



class XEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0):
        # Wrapper to accept list as weight parameter
        weight = torch.Tensor(weight) if weight is not None else weight
        super(XEntropyLoss, self).__init__(weight, size_average, ignore_index,
                                           reduce, reduction, label_smoothing)
