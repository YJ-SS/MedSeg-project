from monai.networks.nets import ResNet, ResNetBlock
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet(
    block=ResNetBlock,
    layers=[2,2,2,2],
    block_inplanes=[64,128,256,512],
    spatial_dims=3,
    n_input_channels=1,
    num_classes=3
).to(device)
x = torch.randn([1,1,32,32,32]).to(device)
y = net(x)
print(y.shape)
