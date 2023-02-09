import monai.networks.nets as mnn
import torch.nn as nn

from networks.core import EnhancedLightningModule



# Give every Monai's model default parameters to ease the build call

class SwinUNETR(EnhancedLightningModule):
    def __init__(self,
                 # EnhancedLightningModule parameters
                 loss=nn.BCELoss(), optimizer={"name": "Adam", "params": {}},
                 lr_scheduler=True, final_activation=nn.Softmax(dim=1), metrics=[],
                 # Monai's SwinUNETR parameters
                 img_size=(256, 256, 256), in_channels=1, out_channels=2,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), feature_size=24,
                 norm_name="instance", drop_rate=0, attn_drop_rate=0,
                 dropout_path_rate=0, normalize=True, use_checkpoint=False,
                 spatial_dims=3, downsample="merging"):
        super(SwinUNETR, self).__init__(
                loss=loss, optimizer=optimizer, lr_scheduler=lr_scheduler,
                final_activation=final_activation, metrics=metrics
                )
        self.model = mnn.SwinUNETR(
                img_size, in_channels, out_channels, depths=depths,
                num_heads=num_heads, feature_size=feature_size,
                norm_name=norm_name, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate, normalize=normalize,
                use_checkpoint=use_checkpoint, spatial_dims=spatial_dims,
                downsample=downsample
                )
        self.out_channels = out_channels

    def forward(self, x):
        return self.model(x)

class UNETR(EnhancedLightningModule):
    def __init__(self,
                 # EnhancedLightningModule parameters
                 loss=nn.BCELoss(), optimizer={"name": "Adam", "params": {}},
                 lr_scheduler=True, final_activation=nn.Softmax(dim=1), metrics=[],
                 # Monai's UNETR parameters
                 in_channels=1, out_channels=2, img_size=(256, 256, 256),
                 feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
                 pos_embed="conv", norm_name="instance", conv_block=True,
                 res_block=True, dropout_rate=0, spatial_dims=3, qkv_bias=False,
                 ):
        super(UNETR, self).__init__(
                loss=loss, optimizer=optimizer, lr_scheduler=lr_scheduler,
                final_activation=final_activation, metrics=metrics
                )
        self.model = mnn.UNETR(
                in_channels, out_channels, img_size, feature_size=feature_size,
                hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads,
                pos_embed=pos_embed, norm_name=norm_name, conv_block=conv_block,
                res_block=res_block, dropout_rate=dropout_rate,
                spatial_dims=spatial_dims, qkv_bias=qkv_bias
                )
        self.out_channels = out_channels

    def forward(self, x):
        return self.model(x)
