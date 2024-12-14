import torch
import torch.nn as nn
from monai.networks.blocks import TransformerBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block = TransformerBlock(
    hidden_size=64,
    mlp_dim=64,
    num_heads=4,
).to(device)
x = torch.randn([1,64,8,8,8]).to(device)
B, C, D, H, W = x.shape
x_reshaped = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
print(x_reshaped.shape)
y = block(x_reshaped)
print(y.shape)
