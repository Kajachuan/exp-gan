import torch
import torch.nn as nn
from typing import Tuple

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, return_indices=False) -> None:
        super(ComplexMaxPool2d, self).__init__()
        self.maxpool_real = nn.MaxPool2d(kernel_size, return_indices=return_indices)
        self.maxpool_imag = nn.MaxPool2d(kernel_size, return_indices=return_indices)
        self.return_indices = return_indices

    def forward(self, z: torch.Tensor):
        if self.return_indices:
            z_real, indices_real = self.maxpool_real(z[...,0])
            z_imag, indices_imag = self.maxpool_imag(z[...,1])
            z = torch.stack((z_real, z_imag), dim=-1)
            indices = torch.stack((indices_real, indices_imag), dim=-1)
            return z, indices
        else:
            return torch.stack((
                self.maxpool_real(z[...,0]), 
                self.maxpool_imag(z[...,1])
            ), dim=-1)

class ComplexMaxUnpool2d(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super(ComplexMaxUnpool2d, self).__init__()
        self.maxunpool_real = nn.MaxUnpool2d(kernel_size)
        self.maxunpool_imag = nn.MaxUnpool2d(kernel_size)

    def forward(
        self, 
        z: torch.Tensor, 
        indices: torch.Tensor, 
        output_size: torch.Size
    ) -> torch.Tensor:
    
        z_real = self.maxunpool_real(z[...,0], indices=indices[...,0], output_size=output_size)
        z_imag = self.maxunpool_imag(z[...,1], indices=indices[...,1], output_size=output_size)
        return torch.stack((z_real, z_imag), dim=-1)

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(ComplexLinear, self).__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imag = nn.Linear(in_features, out_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            self.linear_real(z[...,0]) - self.linear_imag(z[...,1]),
            self.linear_real(z[...,1]) + self.linear_imag(z[...,0])
        ), dim=-1)

class ComplexConv2d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        padding: int = 0
    ) -> None:
        
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, z) -> torch.Tensor:
        return torch.stack((
            self.conv_real(z[...,0]) - self.conv_imag(z[...,1]),
            self.conv_real(z[...,1]) + self.conv_imag(z[...,0])
        ), dim=-1)