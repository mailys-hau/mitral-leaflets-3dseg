import inspect as ispc
import torch.nn as nn

from losses import *
from networks.mixed_architecture import SwinUNETR, UNETR
from networks.unet import UNet, ResUNet




_all_models = {"UNet": UNet, "SwinUNETR": SwinUNETR, "ResUNet": ResUNet,
               "UNETR": UNETR}
_all_losses = {"BalancedEntropyLoss": BalancedEntropyLoss,
               "DiceLoss": DiceLoss, "DiceEntropyLoss": DiceEntropyLoss,
               "DiceFocalLoss": DiceFocalLoss, "FocalLoss": FocalLoss,
               "PreservedBalancedEntropyLoss": PreservedBalancedEntropyLoss,
               "XEntropyLoss": XEntropyLoss,
               "TopkFocalLoss": TopkFocalLoss}



def build_model(name, loss, optimizer, weights=None, **params):
    losses = dict(ispc.getmembers(nn, ispc.isclass))
    losses.update(_all_losses)
    loss = losses[loss.pop("name")](**loss)
    if name not in _all_models:
        raise ValueError(f"Model {name} not found. Chose a model from {list(_all_models.keys())}.")
    if weights:
        print("Loading network weights...")
        net = _all_models[name].load_from_checkpoint(weights, loss=loss,
                                                     optimizer=optimizer, **params)
    else:
        net = _all_models[name](loss=loss, optimizer=optimizer, **params)
    return net
