"""
Metrics computing some type of distances between two sets (aka segmentation
boundaries). `compute` is the same for all so it's in parent's class.
"""
import monai.metrics as mm

from metrics.core import MonaiMetric


#TODO: Handle get_not_nans option from Monai



class HausdorffDistance95(MonaiMetric):
    def __init__(self, include_background=False, distance_metric="euclidean",
                 directed=False, reduction="none", multidim_average="global"):
        super(HausdorffDistance95, self).__init__(include_background,
                                                  distance_metric, reduction,
                                                  multidim_average)
        self.directed = directed
        self.name = "hdf"
        self._create_state()

    def update(self, preds, targets):
        preds, targets = self._tensor_format(preds, targets)
        res = mm.compute_hausdorff_distance(
                y_pred=preds,
                y=targets,
                include_background=self.include_background,
                distance_metric=self.distance_metric,
                percentile=95,
                directed=self.directed
                )
        self._update_state(res.to(self.device))

class SurfaceDistance(MonaiMetric):
    def __init__(self, include_background=False, symmetric=False,
                 distance_metric="euclidean", reduction="none",
                 multidim_average="global"):
        super(SurfaceDistance, self).__init__(include_background, distance_metric,
                                              reduction, multidim_average)
        self.symmetric = symmetric
        self.name = "masd"
        self._create_state()

    def update(self, preds, targets):
        preds, targets = self._tensor_format(preds, targets)
        res = mm.compute_average_surface_distance(
                y_pred=preds,
                y=targets,
                include_background=self.include_background,
                symmetric=self.symmetric,
                distance_metric=self.distance_metric,
                )
        self._update_state(res.to(self.device))
