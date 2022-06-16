import torch
import torch.nn as nn
from typing import Tuple

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, padding: int = 0) -> None:
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_real(z_real) - self.conv_imag(z_imag), \
               self.conv_real(z_imag) + self.conv_imag(z_real)