import monai.networks.layers.simplelayers as mnl

from utils import TensorList



class SkipConnection(mnl.SkipConnection):
    """ Same as monai, but don't post-process x and y. Needed for multi-decoder settings. """
    def __init__(self, submodule, dim=1, mode="cat"):
        super().__init__(submodule, dim, mode)

    def forward(self, x):
        y = self.submodule(x)
        return TensorList(x, y)
