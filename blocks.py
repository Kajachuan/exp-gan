import torch
import torch.nn as nn
from typing import Tuple
from utils import *

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout: float = 0.0, 
        pooling: bool = True
    ) -> None:

        super(EncoderBlock, self).__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, 3, 1)
        self.activation1 = nn.ReLU()
        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, 1)
        self.activation2 = nn.Dropout(dropout) if dropout > 0.0 else nn.ReLU()
        self.maxpool = ComplexMaxPool2d(kernel_size=2, return_indices=True) if pooling else None

    def forward(self, z: torch.Tensor):
        z = self.conv1(z)
        z = self.activation1(z)
        z = self.conv2(z)
        z = self.activation2(z)
        if self.maxpool is not None:
            return self.maxpool(z), z
        else:
            return z

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        self.maxunpool = ComplexMaxUnpool2d(kernel_size=2)
        self.conv1 = ComplexConv2d(in_channels + skip_channels, out_channels, 3, 1)
        self.activation1 = nn.ReLU()
        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, 1)
        self.activation2 = nn.ReLU()

    def forward(
        self, 
        z: torch.Tensor,
        indices: torch.Tensor,
        skip: torch.Tensor,
        output_size: torch.Size
    ) -> torch.Tensor:

        z = self.maxunpool(z, indices, output_size)
        z = torch.cat([z, skip], dim=1)
        z = self.conv1(z)
        z = self.activation1(z)
        z = self.conv2(z)
        z = self.activation2(z)
        return z