""" Monai's metrics don't inherit `torchmetrics.Metric`, we're fixing this """
import torch
import torch.nn.functional as F

from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat



class MonaiMetric(Metric):
    """ We mimic `torchmetrics.classification`, to have per class results """
    is_differentiable = False
    higher_is_better = False
    full_state_update = False # Whether batches are dependant from each other
    def __init__(self, include_background=False, distance_metric="euclidean",
                 reduction="mean", multidim_average="global"):
        super(MonaiMetric, self).__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.average = reduction
        self.multidim_average = multidim_average

    def _create_state(self, size=1):
        if self.multidim_average == "samplewise":
            default = lambda: []
            reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.float)
            reduce_fx = "sum"
        self.add_state(self.name, default=default(), dist_reduce_fx=reduce_fx)

    def _update_state(self, val):
        if self.multidim_average == "samplewise":
            self.__getattribute__(self.name).append(val)
        else:
            prev = self.__getattribute__(self.name)
            setattr(self, self.name, prev + val)

    def _final_state(self):
        """ Final aggregation in case of list states. """
        return dim_zero_cat(self.__getattribute__(self.name))

    def compute(self):
        res = self._final_state()
        sum_dim = None if self.multidim_average == "global" else 1
        if self.average == "micro":
            return res.sum(sum_dim)
        if self.average == "macro":
            return res.float().mean(sum_dim)
        elif self.average == "none" or self.average is None:
            if self.multidim_average == "global":
                return res.mean(0) # Across classes
            return res.mean(1) # Across samples

    def _one_hot(self, x):
        # X is of shape (B,C,H,W,D), one_hot puts C at the end
        idx = x.argmax(dim=1)
        return F.one_hot(idx, num_classes=x.shape[1]).transpose(1, -1)

    def _tensor_format(self, preds, targets):
        preds = self._one_hot(preds) # Targets are already one hot
        preds = preds.int() if preds.is_floating_point() else preds
        targets = targets.int() if targets.is_floating_point() else targets
        return preds, targets
