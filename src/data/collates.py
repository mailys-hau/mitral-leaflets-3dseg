"""
Custom `collate_fn` for PyTorch Dataloader
"""

import torch

from utils import TensorList



def collate_tensorlist(batch):
    # Receive [(input, target), ...]
    inputs = torch.stack([ b[0] for b in batch ])
    tmp = [ b[1] for b in batch ]
    targets = TensorList(*[ torch.stack(elt) for elt in zip(*tmp) ])
    return inputs, targets
