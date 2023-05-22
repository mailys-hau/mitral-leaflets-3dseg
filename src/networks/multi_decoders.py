import monai.networks.nets as mnn
import torch
import torch.nn as nn

from copy import deepcopy
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep

from networks.utils import SkipConnection
from utils import rec_flatten, TensorList



class _UNetnUps(nn.Module):
    """ Create a UNet with a decoder per outputed channels. Modeled after `mnn.BasicUNet` """
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=2,
                 features=(16, 16, 32, 64, 128, 16),
                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                 norm=("instance", {"affine": True}), bias=True, dropout=0, upsample="deconv"):
        super(_UNetnUps, self).__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"UNetnUps features: {fea}")
        self.out_channels = out_channels
        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.upcat_4 = nn.ModuleList()
        self.upcat_3 = nn.ModuleList()
        self.upcat_2 = nn.ModuleList()
        self.upcat_1 = nn.ModuleList()
        self.final_conv = nn.ModuleList()
        for i in range(self.out_channels):
            self.upcat_4.append(UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample))
            self.upcat_3.append(UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample))
            self.upcat_2.append(UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample))
            self.upcat_1.append(UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample))
            self.final_conv.append(Conv["conv", spatial_dims](fea[5], 1, kernel_size=1))

    def forward(self, x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        outs = TensorList()
        for i in range(self.out_channels):
            u4 = self.upcat_4[i](x4, x3)
            u3 = self.upcat_3[i](u4, x2)
            u2 = self.upcat_2[i](u3, x1)
            u1 = self.upcat_1[i](u2, x0)
            outs.append(self.final_conv[i](u1))
        return outs

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
