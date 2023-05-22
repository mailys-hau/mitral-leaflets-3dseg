import monai.networks.nets as mnn
import torch
import torch.nn as nn

from copy import deepcopy
from monai.utils import ensure_tuple_rep

from networks.utils import SkipConnection
from utils import rec_flatten, TensorList



class _ResUNetnUps(mnn.UNet):
    """ Create a ResUNet with a decoder per outputed channels """
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=2,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), kernel_size=3,
                 up_kernel_size=3, num_res_units=2, act="PRELU", norm="INSTANCE",
                 dropout=0, bias=True, adn_ordering="NDA"):
        super(_ResUNetnUps, self).__init__(
                spatial_dims, in_channels, out_channels, channels, strides,
                kernel_size=kernel_size, up_kernel_size=up_kernel_size,
                num_res_units=num_res_units, act=act, norm=norm, dropout=dropout,
                bias=bias, adn_ordering=adn_ordering
                )
        def _create_block(inc, outc, channels, strides, is_top):
            c = channels[0]
            s = strides[0]
            if len(channels) > 2:
                downblock, upblock = _create_block(c, c, channels[1:], strides[1:], False)
                upc = c * 2
            else:
                downblock = self._get_bottom_layer(c, channels[1])
                upblock = None
                upc = c + channels[1]
            down = self._get_down_layer(inc, c, s, is_top)
            if is_top:
                up = self._get_up_layer(upc, 1, s, is_top)
            else:
                up = self._get_up_layer(upc, outc, s, is_top)
            return (self._get_connection_down_block(down, downblock),
                    self._get_connection_up_block(up, upblock))

        self.down, up = _create_block(in_channels, out_channels, self.channels, self.strides, True)
        self.ups = nn.ModuleList([deepcopy(up) for i in range(out_channels)])
        del self.model

    def _get_connection_down_block(self, down_path, subblock):
        return nn.Sequential(down_path, SkipConnection(subblock))

    def _get_connection_up_block(self, up_path, subblock):
        return subblock.append(up_path) if subblock else nn.Sequential(up_path)

    def _up(self, up, latents):
        #FIXME: Enable use of `mode` as in monai's SkipConnection
        # (B, C, W, L, D)
        out = up[0](torch.cat([latents[1], latents[0]], dim=1))
        for k in range(1, len(up)):
            out = up[k](torch.cat([latents[k + 1], out], dim=1))
        return out

    def forward(self, x):
        latents = self.down(x)
        latents = rec_flatten(latents)
        latents.reverse()
        outs = TensorList(*[self._up(up, latents) for up in self.ups])
        return outs
