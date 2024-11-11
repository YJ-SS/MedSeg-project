from typing import Tuple, Any, List
import torch
import torch.nn as nn
from monai.networks.blocks import TransformerBlock

class TransNet(nn.Module):
    def __init__(
            self,
            in_channel: int,
            num_class: int,
            channel_list: List[int],
            device: torch.device,
    ):
        super().__init__()

if __name__ == '__main__':
    x = torch.randn([32,32,1]).cuda()
    block = TransformerBlock(
        hidden_size=4,
        mlp_dim=4,
        num_heads=1,

    ).cuda()
    print(block(x).shape)

