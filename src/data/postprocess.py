"""
Model after `monai.transforms.utils.remove_small_objects`
"""
import numpy as np
import scipy.ndimage as sci
import torch

from monai.transforms.utils_pytorch_numpy_unification import unique
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type




ndims = 3 # We work in 3D
STRUCT1 = sci.generate_binary_structure(ndims, 1) # 1 connectivity
STRUCT1_5 = np.stack([STRUCT1[1]] * 3) # In between
STRUCT3 = sci.generate_binary_structure(ndims, 3) # 3 connectivity

MORPHOLOGIES = {"erosion": sci.grey_erosion, "dilation": sci.grey_dilation,
                "opening": sci.grey_opening, "closing": sci.grey_closing}



def grey_morphology(vol, name, threshold=0.5, ignore_background=True, size=None,
                    footprint=STRUCT1_5, structure=None, mode="reflect", cval=0, origin=0):
    """ Apply given morphology operation and keep everything over threshold """
    # Receive (C, W, H, D) shape, C number of classes
    if len(unique(vol)) == 1:
        return vol, None, None # If all equal to one value, no need to call skimage
    #filtered = torch.where(vol >= threshold, vol, 0)
    vol_np, *_ = convert_data_type(vol, np.ndarray)
    out_np = np.empty_like(vol_np)
    if ignore_background:
        out_np[0] = vol_np[0]
        start = 1
    else:
        start = 0
    morpho = MORPHOLOGIES[name]
    for c in range(start, len(vol_np)): # Classes are stacked
        processed = morpho(vol_np[c], size, footprint, structure, output=None,
                           mode=mode, cval=cval, origin=origin)
        out_np[c] = np.where(processed, np.maximum(threshold, vol_np[c]), 0)
    out, *_ = convert_to_dst_type(out_np, vol)
    return out
