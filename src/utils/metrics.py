""" Need to overwrite Monai's metrics since they're not torchmetrics.Metric """
import monai.metrics as mm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import tensor
from torchmetrics import Metric



#TODO: Generalize + code refactor

class HausdorffDistance95(Metric):
    is_differentiable = False
    higher_is_better = False
    # Whether batches are independant from each other
    full_state_update = False
    def __init__(self, include_background=False, distance_metric="euclidean",
                 directed=False, reduction="mean", get_not_nans=False):
        super(HausdorffDistance95, self).__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.directed = directed
        #FIXME: Accept all type of reduction as defined by MONAI + handle nans
        #self.reduction = reduction
        #self.get_not_nans = get_not_nans
        self.add_state("sum_hsd", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def _one_hot(self, x):
        # X is of shape (B,C,H,W,D), one_hot puts C at the end
        idx = x.argmax(dim=1)
        return F.one_hot(idx, num_classes=x.shape[1]).transpose(1, -1)

    def update(self, preds, target):
        preds = self._one_hot(preds)
        preds = preds if preds.is_floating_point else preds.float()
        target = target if target.is_floating_point else target.float()
        res = mm.compute_hausdorff_distance(
                y_pred=preds,
                y=target,
                include_background=self.include_background,
                distance_metric=self.distance_metric,
                percentile=95,
                directed=self.directed
                )
        self.sum_hsd += torch.sum(res)
        self.total += res.numel()


    def compute(self):
        return self.sum_hsd / self.total

class SurfaceDistance(Metric):
    is_differentiable = False
    higher_is_better = False
    # Whether batches are independant from each other
    full_state_update = False

    def __init__(self, include_background=False, symmetric=False,
                 distance_metric="euclidean", reduction="mean",
                 get_not_nans=False):
        super(SurfaceDistance, self).__init__()
        self.include_background = include_background
        self.symmetric = symmetric
        self.distance_metric = distance_metric
        #FIXME: Accept all type of reduction as defined by MONAI + handle nans
        #self.reduction = reduction
        #self.get_not_nans = get_not_nans
        self.add_state("sum_masd", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def _one_hot(self, x):
        # X is of shape (B,C,H,W,D), one_hot puts C at the end
        idx = x.argmax(dim=1)
        return F.one_hot(idx, num_classes=x.shape[1]).transpose(1, -1)

    def update(self, preds, target):
        preds = self._one_hot(preds)
        preds = preds if preds.is_floating_point else preds.float()
        target = target if target.is_floating_point else target.float()
        res = mm.compute_average_surface_distance(
                y_pred=preds,
                y=target,
                include_background=self.include_background,
                symmetric=self.symmetric,
                distance_metric=self.distance_metric,
                )
        self.sum_masd += torch.sum(res)
        self.total += res.numel()


    def compute(self):
        return self.sum_masd / self.total
