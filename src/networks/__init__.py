import inspect as ispc
import torch.nn as nn


from networks.unet import BasicUNet, UNet




_all_models = {"BasicUNet": BasicUNet, "UNet": UNet}



def build_model(name, loss, optimizer, weights=None, **params):
    losses = dict(ispc.getmembers(nn, ispc.isclass))
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
