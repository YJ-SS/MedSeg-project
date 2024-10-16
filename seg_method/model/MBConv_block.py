import torch
import torch.nn as nn

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, groups=in_channels, padding=kernel_size//2)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AttentionModule3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class MBConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.depthwise_separable = DepthwiseSeparableConv3D(in_channels, out_channels)
        self.attention = AttentionModule3D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_separable(x)
        x = self.attention(x)
        x = self.relu(x)
        return x
if __name__ == '__main__':

    # Example usage
    x = torch.randn(1, 1, 16, 16, 16)  # 3D input (batch_size, channels, depth, height, width)
    mbconv_3d = MBConvBlock3D(1, 64)
    x = x.cuda()
    mbconv_3d = mbconv_3d.cuda()
    output = mbconv_3d(x)
    print(output.shape)  # torch.Size([1, 64, 16, 16, 16])
