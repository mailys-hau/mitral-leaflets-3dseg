import monai.transforms as mt

from monai.utils import Method, PytorchPadMode




NORMS = {"256": mt.NormalizeIntensity(subtrahend=0, divisor=255),
         "std": mt.NormalizeIntensity()}



class ResizeWithPadOrCropd(mt.ResizeWithPadOrCropd):
    def __init__(self, keys, spatial_size, method=Method.SYMMETRIC,
                 mode=PytorchPadMode.CONSTANT, **pad_kwargs):
        pad_kwargs.pop("multiclass", None) # This is why we overwrite
        super(ResizeWithPadOrCropd, self).__init__(
                keys, spatial_size, method=method, mode=mode, **pad_kwargs)


class ResizeWithPadOrRandCropd(mt.ResizeWithPadOrCropd):
    def __init__(self, keys, spatial_size, method=Method.SYMMETRIC,
                 mode=PytorchPadMode.CONSTANT, **pad_kwargs):
        pad_kwargs.pop("multiclass", None)
        super(ResizeWithPadOrRandCropd, self).__init__(
                keys, spatial_size, method=method, mode=mode, **pad_kwargs)
        # Overwrite cropper
        self.cropper = mt.RandSpatialCropd(keys, roi_size=spatial_size, random_center=False)

class ResizeWithPadOrCenterRandCropd(mt.ResizeWithPadOrCropd):
    def __init__(self, keys, spatial_size, method=Method.SYMMETRIC,
                 mode=PytorchPadMode.CONSTANT, **pad_kwargs):
        pad_kwargs.pop("multiclass", None)
        super(ResizeWithPadOrCenterRandCropd, self).__init__(
                keys, spatial_size, method=method, mode=mode, **pad_kwargs)
        # Overwrite cropper
        self.cropper = mt.RandSpatialCropd(keys, roi_size=spatial_size)

class ResizeWithPadOrRandCropByLabelClassesd(mt.ResizeWithPadOrCropd):
    def __init__(self, keys, spatial_size, multiclass=False, method=Method.SYMMETRIC,
                 mode=PytorchPadMode.CONSTANT, **pad_kwargs):
        # Ensure crop is centered around a leaflet voxel
        ratio = [0, 1, 1] if multiclass else [0, 1]
        pad_kwargs.pop("multiclass", None)
        super(ResizeWithPadOrRandCropByLabelClassesd, self).__init__(
                keys, spatial_size, method=method, mode=mode, **pad_kwargs)
        # Overwrite cropper
        self.cropper = mt.RandCropByLabelClassesd(keys, keys[-1], spatial_size,
                                                  ratios=ratio, num_samples=len(ratio))




RESIZE = {"by-classes": ResizeWithPadOrRandCropByLabelClassesd,
          "center": ResizeWithPadOrCropd,
          "center-random": ResizeWithPadOrCenterRandCropd,
          "random": ResizeWithPadOrRandCropd}
