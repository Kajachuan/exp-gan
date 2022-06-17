import torch
import torch.nn as nn
from typing import Tuple
from utils import *

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 dropout: float = 0.0, pooling: bool = True) -> None:
        super(EncoderBlock, self).__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, 3, 1)
        self.activation1 = ComplexReLU()
        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, 1)
        self.activation2 = ComplexDropout(dropout) if dropout > 0.0 else ComplexReLU()
        self.maxpool = ComplexMaxPool2d(kernel_size=2, return_indices=True) if pooling else None

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor):
        z_real, z_imag = self.conv1(z_real, z_imag)
        z_real, z_imag = self.activation1(z_real, z_imag)
        z_real, z_imag = self.conv2(z_real, z_imag)
        z_real, z_imag = self.activation2(z_real, z_imag)
        if self.maxpool:
            return self.maxpool(z_real, z_imag), (z_real, z_imag)
        else:
            return z_real, z_imag

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        self.maxunpool = ComplexMaxUnpool2d(kernel_size=2)
        self.conv1 = ComplexConv2d(in_channels + skip_channels, out_channels, 3, 1)
        self.activation1 = ComplexReLU()
        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, 1)
        self.activation2 = ComplexReLU()

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor,
                indices_real: torch.Tensor, indices_imag: torch.Tensor,
                skip_real: torch.Tensor, skip_imag: torch.Tensor,
                output_size: torch.Size):

        z_real, z_imag = self.maxunpool(z_real, z_imag, 
                                        indices_real, indices_imag,
                                        output_size)
        z_real = torch.cat([z_real, skip_real], dim=1)
        z_imag = torch.cat([z_imag, skip_imag], dim=1)
        z_real, z_imag = self.conv1(z_real, z_imag)
        z_real, z_imag = self.activation1(z_real, z_imag)
        z_real, z_imag = self.conv2(z_real, z_imag)
        z_real, z_imag = self.activation2(z_real, z_imag)
        return z_real, z_imag