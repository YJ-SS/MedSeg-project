import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.15):
        super(MultiScaleConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv_fusion = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(drop_prob)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_concat = torch.cat((x1, x2, x3), dim=1)
        x_fused = F.relu(self.conv_fusion(x_concat))
        x_fused = self.dropout(x_fused)
        return x_fused

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn([1,1,32,32,32]).to(device)
    block = MultiScaleConv3D(in_channels=1, out_channels=8)
    y = block(x)
    print(y.shape)