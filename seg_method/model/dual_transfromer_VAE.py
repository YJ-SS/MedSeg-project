import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import PatchMerging, BasicLayer


class transNet(nn.Module):
    def __init__(
            self,
            in_channels,
            num_class,
            embed_dim,
            patch_size,
            window_size,
    ):
        super(transNet, self).__init__()
        patch_size = ensure_tuple_rep(patch_size, dim=3)
        window_size = ensure_tuple_rep(window_size, dim=3)
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[2, 4, 8, 16],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            downsample=PatchMerging
        )
        self.mu_layer = BasicLayer(
            dim=embed_dim * 2**4,
            num_heads=16,
            depth=1,
            window_size=window_size,
            drop_path=[0.1]
        )
        self.var_layer = BasicLayer(
            dim=embed_dim * 2**4,
            num_heads=16,
            depth=1,
            window_size=window_size,
            drop_path=[0.1]
        )
        self.up_layers4 = nn.Sequential()
        self.up_layers3 = nn.Sequential()
        self.up_layers2 = nn.Sequential()
        self.up_layers1 = nn.Sequential()
        for i_up_layer in range(4, 0, -1):
            if i_up_layer == 4:
                layer = nn.ConvTranspose3d(
                    in_channels=embed_dim * 2**i_up_layer,
                    out_channels=embed_dim * 2**(i_up_layer - 1),
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2)
                )
                self.up_layers4.append(layer)
            else:
                channel_reduce_layer = nn.Conv3d(
                    in_channels=embed_dim * 2**(i_up_layer + 1),
                    out_channels=embed_dim * 2**i_up_layer,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1)
                )
                trans_layer = BasicLayer(
                    dim=embed_dim * 2**(i_up_layer),
                    num_heads=2**i_up_layer,
                    depth=2,
                    window_size=window_size,
                    drop_path=[0.1, 0.1],
                    downsample=None
                )
                up_sample_layer = nn.ConvTranspose3d(
                    in_channels=embed_dim * 2**i_up_layer,
                    out_channels=embed_dim * 2**(i_up_layer - 1),
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2)
                )
                if i_up_layer == 3:
                    self.up_layers3.append(channel_reduce_layer)
                    self.up_layers3.append(trans_layer)
                    self.up_layers3.append(up_sample_layer)
                elif i_up_layer == 2:
                    self.up_layers2.append(channel_reduce_layer)
                    self.up_layers2.append(trans_layer)
                    self.up_layers2.append(up_sample_layer)
                elif i_up_layer == 1:
                    self.up_layers1.append(channel_reduce_layer)
                    self.up_layers1.append(trans_layer)
                    self.up_layers1.append(up_sample_layer)

        self.recon_proj = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=embed_dim * 2,
                out_channels=embed_dim * 2,
                kernel_size=(2, 2, 2),
                stride=(2, 2, 2)
            ),
            nn.Conv3d(
                in_channels=embed_dim * 2,
                out_channels=in_channels,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1)
            )
        )

        self.seg_proj = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=embed_dim * 2,
                out_channels=embed_dim * 2,
                kernel_size=(2, 2, 2),
                stride=(2, 2, 2)
            ),
            nn.Conv3d(
                in_channels=embed_dim * 2,
                out_channels=num_class,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1)
            )
        )





    def reparameter(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())
        mu = self.mu_layer(x_out[-1])
        var = self.var_layer(x_out[-1])
        latent = self.reparameter(mu, var)
        x_up3 = self.up_layers4(latent)
        x_up3 = torch.concat([x_up3, x_out[3]], dim=1)
        # print("x_up3 shape=", x_up3.shape)

        x_up2 = self.up_layers3(x_up3)
        x_up2 = torch.concat([x_up2, x_out[2]], dim=1)
        # print("x_up2 shape=", x_up2.shape)

        x_up1 = self.up_layers2(x_up2)
        x_up1 = torch.concat([x_up1, x_out[1]], dim=1)
        # print("x_up1 shape=", x_up1.shape)

        x_up0 = self.up_layers1(x_up1)
        x_up0 = torch.concat([x_up0, x_out[0]], dim=1)
        # print("x_up0 shape=", x_up0.shape)
        recon = self.recon_proj(x_up0)
        # print("recon shape=", recon.shape)
        seg = self.seg_proj(x_up0)
        # print("seg shape=", seg.shape)
        return seg, recon, mu, var, latent

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn([1,1,224,192,160]).to(device)
    net = transNet(
        in_channels=1,
        num_class=5,
        embed_dim=8,
        patch_size=2,
        window_size=7,
    ).to(device)
    y = net(x)
    for i in range(len(y)):
        print("x{0} shape=".format(i), y[i].shape)
