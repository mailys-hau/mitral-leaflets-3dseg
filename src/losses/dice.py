import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.modules.loss._Loss):
    """ Dice score or F1 Score, to be use as a loss """
    def __init__(self, ignore_background=True, zero_div_eps=1e-5):
        super(DiceLoss, self).__init__()
        self.ignore_background = ignore_background
        self.zero_div_eps = zero_div_eps

    def forward(self, pred, target):
        # Receive tensors of shape (B, C,*) | Target is assumed to be one hot
        dim = list(range(2, pred.ndim)) # Don't reduce over batch or classes
        y_pred = F.softmax(pred, dim=1)
        if self.ignore_background and y_pred.shape[1] > 1:
            # If binary segmentation there's nothing to ignore
            # Background is assumed to be first channel
            y_pred, target = y_pred[:,1:], target[:,1:]
        inter = (y_pred * target).sum(dim=dim)
        tpos, ppos = target.sum(dim=dim), y_pred.sum(dim=dim)
        dice = 2 * inter / (tpos + ppos + self.zero_div_eps)
        # Return 1 - dice so we can minimize the loss, and it's not always 0
        return (1 - dice).mean()
