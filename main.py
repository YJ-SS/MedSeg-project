from monai.networks.nets import UNet, SwinUNETR
import torch

# for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
net = SwinUNETR(
    img_size=(160, 192, 224),
    in_channels=1,
    out_channels=4,
    feature_size=24,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24]
).cuda()
x = torch.randn([1,1, 192, 160, 224]).cuda()
y = net(x)
print(y.shape)