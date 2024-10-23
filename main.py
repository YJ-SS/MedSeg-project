from monai.networks.nets import UNet
import torch

net = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=[16, 32, 64, 128],
    strides=[1, 1, 1]
).cuda()
x = torch.randn([1,1,32,32,32]).cuda()
y = net(x)
print(y.shape)