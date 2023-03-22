"""
Modeled after `monai.transforms.utils.remove_small_objects`
"""
import numpy as np
import torch

from monai.transforms.utils_pytorch_numpy_unification import unique
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from scipy.ndimage import binary_fill_holes, grey_closing



def close_and_fill(vol, size=(3, 3, 3), footprint=None, structure=None,
                   mode="reflect", cval=0, origin=0, threshold=0.5,
                   ignore_background=True):
    if len(unique(vol)) == 1:
        return vol # If all equal to one value, no need to call skimage
    filtered = torch.where(vol >= threshold, vol, 0)
    vol_np, *_ = convert_data_type(filtered, np.ndarray)
    out_np = np.empty_like(vol_np)
    if ignore_background:
        out_np[0] = vol_np[0]
        start = 1
    else:
        start = 0
    for c in range(start, len(vol_np)): # Classes are stacked
        closed = grey_closing(vol_np[c], size, footprint, structure, mode=mode,
                              cval=cval, origin=origin)
        filled = binary_fill_holes(vol_np[c], structure, origin=origin)
        out_np[c] = np.where(filled, np.maximum(threshold, vol_np[c]), 0)
    out, *_ = convert_to_dst_type(out_np, vol)
    return out


def closing(vol, size=(3, 3, 3), footprint=None, structure=None, mode="reflect",
            cval=0, origin=0, ignore_background=True):
    if len(unique(vol)) == 1:
        return vol # If all equal to one value, no need to call skimage
    vol_np, *_ = convert_data_type(vol, np.ndarray)
    out_np = np.empty_like(vol_np)
    if ignore_background:
        out_np[0] = vol_np[0]
        start = 1
    else:
        start = 0
    for c in range(start, len(vol_np)): # Classes are stacked
        out_np[c] = grey_closing(vol_np[c], size, footprint, structure,
                                 mode=mode, cval=cval, origin=origin)
    out, *_ = convert_to_dst_type(out_np, vol)
    return out

def fill_holes(vol, structure=None, output=None, origin=0,
               ignore_background=True):
    if len(unique(vol)) == 1:
        return vol # If all equal to one value, no need to call skimage
    vol_np, *_ = convert_data_type(vol, np.ndarray)
    out_np = np.empty_like(vol_np)
    if ignore_background:
        out_np[0] = vol_np[0]
        start = 1
    else:
        start = 0
    for c in range(start, len(vol_np)): # Classes are stacked
        # Return a one hot
        out_np[c] = binary_fill_holes(vol_np[c], structure, origin=origin)
    out, *_ = convert_to_dst_type(out_np, vol)
    return out
