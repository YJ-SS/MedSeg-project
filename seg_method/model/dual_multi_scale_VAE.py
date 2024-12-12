import sys
from typing import List

sys.path.append("..")
import torch
import torch.nn as nn
from multi_scale_conv import MultiScaleConv3D

class MultiScaleNet(nn.Module):
    def __init__(
            self,
            in_channel: int,
            num_class: int,
            channel_list: List[int],
            device: torch.device
    ):
        super(MultiScaleNet, self).__init__()
        self.encode_blocks = []
        self.decode_blocks = []
        self.encode_outputs = []
        self.down_sample = nn.MaxPool3d(kernel_size=2, stride=2, padding=0).to(device)
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear').to(device)
        self.encode_blocks.append(
            nn.Sequential(
                MultiScaleConv3D(
                    in_channels=in_channel,
                    out_channels=channel_list[0]
                ),
                nn.BatchNorm3d(channel_list[0]),
                MultiScaleConv3D(
                    in_channels=channel_list[0],
                    out_channels=channel_list[0]
                )
            )
        )
        for i in range(len(channel_list) - 1):
            self.encode_blocks.append(
                nn.Sequential(
                    MultiScaleConv3D(
                        in_channels=channel_list[i],
                        out_channels=channel_list[i + 1]
                    ),
                    nn.BatchNorm3d(channel_list[i + 1]),
                    MultiScaleConv3D(
                        in_channels=channel_list[i + 1],
                        out_channels=channel_list[i + 1]
                    )
                )
            )
        for i, block in enumerate(self.encode_blocks):
            self.add_module('encode block{0}'.format(i), block)

        self.decode_blocks.append(
            nn.Sequential(
                MultiScaleConv3D(
                    in_channels=channel_list[-1],
                    out_channels=channel_list[-2]
                )
            )
        )

        for i in range(len(channel_list) - 1, 0, -1):
            self.decode_blocks.append(
                nn.Sequential(
                    MultiScaleConv3D(
                        in_channels=channel_list[i],
                        out_channels=channel_list[i - 1]
                    ),
                    nn.BatchNorm3d(channel_list[i - 1]),
                    MultiScaleConv3D(
                        in_channels=channel_list[i - 1],
                        out_channels=channel_list[i - 2] if i - 1 > 0 else channel_list[0]
                    )

                )
            )
        for i, block in enumerate(self.decode_blocks):
            self.add_module('decode block{0}'.format(i), block)

        self.mu_conv = MultiScaleConv3D(
            in_channels=channel_list[-1],
            out_channels=channel_list[-1]
        )
        self.var_conv = MultiScaleConv3D(
            in_channels=channel_list[-1],
            out_channels=channel_list[-1]
        )
        self.seg_head = nn.Sequential(
            MultiScaleConv3D(
                in_channels=channel_list[0],
                out_channels=channel_list[0]
            ),
            nn.BatchNorm3d(channel_list[0]),
            nn.Conv3d(
                in_channels=channel_list[0],
                out_channels=num_class,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self.recon_head = nn.Sequential(
            MultiScaleConv3D(
                in_channels=channel_list[0],
                out_channels=in_channel
            )
        )

    def reparameterize(self, mu, logvar, is_train):
        '''
        在进行训练时引入特征级别噪音
        在验证时不引入噪音
        :param mu:
        :param logvar:
        :param is_train:
        :return:
        '''
        std = torch.exp(0.5 * logvar)
        if is_train:
            eps = torch.rand_like(std)
            return mu + eps * std
        else:
            return mu + std

    def forward(self, x, is_train=True):
        self.encode_outputs = []
        for i, block in enumerate(self.encode_blocks):
            if i < len(self.encode_blocks) - 1:
                x = block(x)
                self.encode_outputs.append(x)
                # print(x.shape)
                x = self.down_sample(x)
            else:
                x = block(x)

        mu = self.mu_conv(x)
        logvar = self.var_conv(x)
        latent = self.reparameterize(mu, logvar, is_train)
        x = latent

        self.encode_outputs.reverse()

        for i, block in enumerate(self.decode_blocks):
            if i == 0:
                x = block(x)
            else:
                x = self.up_sample(x)
                x = torch.concat([self.encode_outputs[i - 1], x], dim = 1)
                x = block(x)
        pre_seg = self.seg_head(x)
        recon = self.recon_head(x)
        return pre_seg, recon, mu, logvar, latent


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn([2,1,32,32,32]).to(device)
    net = MultiScaleNet(
        in_channel=1,
        num_class=5,
        channel_list=[16, 32, 64, 128],
        device=device
    )
    pre_seg, recon, mu, logvar, latent = net(x)
    print(pre_seg.shape, recon.shape, mu.shape, logvar.shape, latent.shape)

















