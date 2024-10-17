import sys
sys.path.append("..")
from typing import Tuple, Any, List
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from MBConv_block import MBConvBlock3D

class MBConvNet(nn.Module):
    def __init__(
            self,
            in_channel: int,
            num_class: int,
            channel_list: List[int],
            residual: bool,
            MBConv: bool,
            device: torch.device,
    ):
        super().__init__()
        # valid_devices = {'cuda', 'cpu'}
        # if device not in valid_devices:
        #     raise ValueError(f"Invalid value for device. Expected one of {valid_devices}, but got {device}")

        self.residual = residual
        self.encode_blocks = []
        self.decode_blocks = []
        self.encode_outputs = []
        self.down_sample = nn.MaxPool3d(kernel_size=2, stride=2, padding=0).to(device)
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear').to(device)

        self.add_module('down_sample_layer', self.down_sample)
        self.add_module('up_sample_layer', self.up_sample)

        # Build encoder

        # Use conventional convolution method
        if self.residual:
            self.encode_blocks.append(
                nn.Sequential(
                    ResidualUnit(
                        spatial_dims=3,
                        in_channels=in_channel,
                        out_channels=channel_list[0],
                        kernel_size=3,
                        strides=1
                    ).to(device)
                )
            )
        else:
            self.encode_blocks.append(
                nn.Sequential(
                    Convolution(
                        spatial_dims=3,
                        in_channels=in_channel,
                        out_channels=channel_list[0],
                        kernel_size=3,
                        strides=1
                    ).to(device),

                    Convolution(
                        spatial_dims=3,
                        in_channels=channel_list[0],
                        out_channels=channel_list[0],
                        kernel_size=3,
                        strides=1
                    ).to(device),

                )
            )

        for i in range(0, len(channel_list) - 1):
            if i < len(channel_list) - 2:
                # Not last layer (bottleneck)
                # Use conventional convolution
                if self.residual:
                    self.encode_blocks.append(
                        ResidualUnit(
                            spatial_dims=3,
                            in_channels=channel_list[i],
                            out_channels=channel_list[i + 1],
                            kernel_size=3,
                            strides=1
                        ).to(device)
                    )
                else:
                    self.encode_blocks.append(
                        nn.Sequential(
                            Convolution(
                                spatial_dims=3,
                                in_channels=channel_list[i],
                                out_channels=channel_list[i + 1],
                                kernel_size=3,
                                strides=1
                            ).to(device),

                            Convolution(
                                spatial_dims=3,
                                in_channels=channel_list[i + 1],
                                out_channels=channel_list[i + 1],
                                kernel_size=3,
                                strides=1
                            ).to(device)
                        )
                    )
            else:
                # Last layer(bottle neck): channel: 256->512->256
                self.encode_blocks.append(
                    nn.Sequential(
                        Convolution(
                            spatial_dims=3,
                            in_channels=channel_list[i],
                            out_channels=channel_list[i + 1],
                            kernel_size=3,
                            strides=1
                        ).to(device)
                    )
                )

        for i, block in enumerate(self.encode_blocks):
            self.add_module('encode_block{0}'.format(i), block)

        if MBConv:
            self.mu_fc = MBConvBlock3D(
                in_channels=channel_list[-1],
                out_channels=channel_list[-1]
            ).to(device)
        else:
            self.mu_fc = nn.Conv3d(
                in_channels=channel_list[-1],
                out_channels=channel_list[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            ).to(device)

        if MBConv:
            self.var_fc = MBConvBlock3D(
                in_channels=channel_list[-1],
                out_channels=channel_list[-1]
            ).to(device)
        else:
            self.var_fc = nn.Conv3d(
                in_channels=channel_list[-1],
                out_channels=channel_list[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ).to(device)

        # Build decoder
        self.decode_blocks.append(
            nn.Sequential(
                Convolution(
                    spatial_dims=3,
                    in_channels=channel_list[-1],
                    out_channels=channel_list[-2],
                    kernel_size=3,
                    strides=2,
                    is_transposed=True
                ).to(device)
            )
        )

        for i in range(len(channel_list) - 1, 0, -1):

            self.decode_blocks.append(
                nn.Sequential(
                    Convolution(
                        spatial_dims=3,
                        in_channels=channel_list[i],
                        out_channels=channel_list[i - 1],
                        kernel_size=3,
                        strides=1,
                    ).to(device),
                    # Final block do not up-sample, if i - 1 == 0, then reach the final block
                    Convolution(
                        spatial_dims=3,
                        in_channels=channel_list[i - 1],
                        out_channels=channel_list[i - 2] if i - 1 > 0 else channel_list[0],
                        kernel_size=3,
                        strides=2 if i - 1 > 0 else 1,
                        is_transposed=True if i - 1 > 0 else False
                    ).to(device)
                )
            )

        for i, block in enumerate(self.decode_blocks):
            self.add_module('decode_block{0}'.format(i), block)

        # Segmentation head
        self.seg_head = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=channel_list[0],
                out_channels=channel_list[0],
                kernel_size=3,
                strides=1
            ).to(device),

            nn.Conv3d(
                in_channels=channel_list[0],
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0

            ).to(device)
        )

        # Reconstruction head
        self.recon_head = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=channel_list[0],
                out_channels=1,
                kernel_size=3,
                strides=1
            ).to(device),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x) -> Tuple[Any, Any, Any, Any, Any]:

        self.encode_outputs = []

        for i, block in enumerate(self.encode_blocks):
            if i < len(self.encode_blocks) - 1:
                x = block(x)
                self.encode_outputs.append(x)
                # print(x.shape)
                x = self.down_sample(x)
            else:
                # Last layer in encoder, do not downsample
                x = block(x)
                # print(x.shape)

        mu = self.mu_fc(x)
        logvar = self.var_fc(x)
        latent = self.reparameterize(mu=mu, logvar=logvar)
        x = latent

        self.encode_outputs.reverse()

        for i, block in enumerate(self.decode_blocks):
            if i == 0:
                x = block(x)
            else:
                # print(x.shape, self.encode_outputs[i - 1].shape)
                x = torch.concat([self.encode_outputs[i - 1], x], dim=1)
                x = block(x)

        pre_seg = self.seg_head(x)
        recon = self.recon_head(x)

        return pre_seg, recon, mu, logvar, latent


if __name__ == '__main__':
    net = MBConvNet(
        in_channel=1,
        num_class=5,
        channel_list=[64, 128, 256, 512],
        residual=True,
        device='cpu',
        MBConv=True
    )
    x = torch.randn(([1,1,32, 32, 32]))

    with torch.no_grad():
        pre_label, recon, mu, logvar, latent = net(x)
        print(pre_label.shape)






