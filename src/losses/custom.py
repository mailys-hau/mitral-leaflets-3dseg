import torch
import torch.nn as nn

from losses.dice import DiceLoss
from losses.x_entropy import FocalLoss



class DiceEntropyLoss(nn.modules.loss._WeightedLoss):
    """ Combine Dice score and CrossEntropy as loss """
    def __init__(self, weight=None,
                 # Dice parameters
                 ignore_background=True, zero_div_eps=1e-5,
                 # CrossEntropyLoss parameters
                 entropy_weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0):
        weight = torch.Tensor(weight) if weight is not None else torch.Tensor([1, 1])
        entropy_weight = torch.Tensor(weight) if entropy_weight is not None\
                            else entropy_weight
        super(DiceEntropyLoss, self).__init__(weight)
        self.dice_loss = DiceLoss(ignore_background, zero_div_eps)
        self.xentropy_loss = nn.CrossEntropyLoss(entropy_weight, size_average,
                                                 ignore_index, reduce, reduction,
                                                 label_smoothing)

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        xentropy = self.xentropy_loss(pred, target)
        return self.weight[0] * dice + self.weight[1] * xentropy

class DiceFocalLoss(nn.modules.loss._WeightedLoss):
    """ Combine Dice score and CrossEntropy as loss """
    def __init__(self, weight=None,
                 # Dice parameters
                 ignore_background=True, zero_div_eps=1e-5,
                 # FocalLoss parameters
                 alpha=0.25, gamma=2, size_average=None, ignore_index=-100,
                 reduce=None, reduction="mean", label_smoothing=0):
        weight = torch.Tensor(weight) if weight is not None else torch.Tensor([1, 1])
        super(DiceFocalLoss, self).__init__(weight)
        self.dice_loss = DiceLoss(ignore_background, zero_div_eps)
        self.focal_loss = FocalLoss(alpha, gamma, size_average, ignore_index,
                                    reduce, reduction, label_smoothing)

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.weight[0] * dice + self.weight[1] * focal

