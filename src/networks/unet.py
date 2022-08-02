"""
All `_Monai*` are wrapper around Monai networks that just add the possibility to pass
any arguments to the init function. This allows to use multi-inheritance with super.
Multi-inheritance allows to take advantage of existing networks' implementation and
pytorch lighning module.
"""
import monai.networks.nets as mnn
import torch.nn as nn

from networks.core import EnhancedLightningModule



class _MonaiBasicUNet(mnn.BasicUNet):
    # Wrapper around Monai BasicUNet. Accept **kwargs for multi-inheritance
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=2,
                 features=(32, 32, 64, 128, 256, 32),
                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm=("instance", {"affine": True}), bias=True, dropout=0.0,
                 upsample="deconv", **kwargs):
        super(_MonaiBasicUNet, self).__init__(spatial_dims, in_channels, out_channels,
                                              features, act, norm, bias, dropout,
                                              upsample)

    def forward_right(self, x):
        return super(_MonaiBasicUNet, self).forward(x)

class _MonaiUNet(mnn.UNet):
    # Wrapper around Monai UNet. Accept **kwargs for multi-inheritance
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides,
                 kernel_size=3, up_kernel_size=3, num_res_units=0, act="PRELU",
                 norm="INSTANCE", dropout=0.0, bias=True, adn_ordering="NDA", **kwargs):
        super(_MonaiUNet, self).__init__(spatial_dims, in_channels, out_channels,
                                         channels, strides, kernel_size, up_kernel_size,
                                         num_res_units, act, norm, dropout, bias,
                                         adn_ordering)

    def forward_right(self, x):
        return super(_MonaiUNet, self).forward(x)


class BasicUNet(EnhancedLightningModule, _MonaiBasicUNet):
    def __init__(self, loss=nn.CrossEntropyLoss(),
                 optimizer={"name": "Adam", "params": {}}, metrics=[], **unet_kwargs):
        # /!\ Arguments must have the same name as in mother classes /!\
        super(BasicUNet, self).__init__(
                # BasicUNet parameters
                **unet_kwargs,
                # EnhancedLightningModule parameters
                loss=loss, optimizer=optimizer, metrics=metrics
                )

    # Inheritance is resolved from left to right parent and both have `forward` method
    def forward_right(self, x):
        return super(BasicUNet, self).forward_right(x)

    # Need to be define because abstract in parent
    def forward(self, x):
        return super(BasicUNet, self).forward(x)

class UNet(EnhancedLightningModule, _MonaiUNet):
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=2,
                 channels=(8, 16, 32), strides=(2, 2), loss=nn.CrossEntropyLoss(),
                 optimizer={"name": "Adam", "params": {}}, metrics=[], **unet_kwargs):
        # /!\ Arguments must have the same name as in mother classes /!\
        super(UNet, self).__init__(
                # UNet parameters
                spatial_dims, in_channels, out_channels, channels, strides, **unet_kwargs,
                # EnhancedLightningModule parameters
                loss=loss, optimizer=optimizer, metrics=metrics
                )

    # Inheritance is resolved from left to right parent and both have `forward` method
    def forward_right(self, x):
        return super(UNet, self).forward_right(x)

    # Need to be define because abstract in parent
    def forward(self, x):
        return super(UNet, self).forward(x)
