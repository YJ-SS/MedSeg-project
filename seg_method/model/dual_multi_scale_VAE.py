import sys
from typing import List

sys.path.append("..")
import torch
import torch.nn as nn

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
