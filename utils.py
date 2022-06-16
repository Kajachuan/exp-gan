import torch
import torch.nn as nn
from typing import Tuple

class ComplexReLU(nn.Module):
    def __init__(self) -> None:
        super(ComplexReLU, self).__init__()
        self.relu_real = nn.ReLU()
        self.relu_imag = nn.ReLU()

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.relu_real(z_real), self.relu_imag(z_imag)

class ComplexSigmoid(nn.Module):
    def __init__(self) -> None:
        super(ComplexSigmoid, self).__init__()
        self.sigmoid_real = nn.Sigmoid()
        self.sigmoid_imag = nn.Sigmoid()

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sigmoid_real(z_real), self.sigmoid_imag(z_imag)

class ComplexDropout(nn.Module):
    def __init__(self, p: float) -> None:
        super(ComplexDropout, self).__init__()
        self.dropout_real = nn.Dropout(p)
        self.dropout_imag = nn.Dropout(p)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dropout_real(z_real), self.dropout_imag(z_imag)

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, return_indices=False) -> None:
        super(ComplexMaxPool2d, self).__init__()
        self.maxpool_real = nn.MaxPool2d(kernel_size, return_indices=return_indices)
        self.maxpool_imag = nn.MaxPool2d(kernel_size, return_indices=return_indices)
        self.return_indices = return_indices

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor):
        if self.return_indices:
            z_real, indices_real = self.maxpool_real(z_real)
            z_imag, indices_imag = self.maxpool_imag(z_imag)
            return (z_real, z_imag), (indices_real, indices_imag)
        else:
            return self.maxpool_real(z_real), self.maxpool_imag(z_imag)

class ComplexMaxUnpool2d(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super(ComplexMaxUnpool2d, self).__init__()
        self.maxunpool_real = nn.MaxUnpool2d(kernel_size)
        self.maxunpool_imag = nn.MaxUnpool2d(kernel_size)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor, 
                indices_real: torch.Tensor, indices_imag: torch.Tensor,
                output_size: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
            z_real = self.maxunpool_real(z_real, indices=indices_real, output_size=output_size)
            z_imag = self.maxunpool_imag(z_imag, indices=indices_imag, output_size=output_size)
            return z_real, z_imag

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(ComplexLinear, self).__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imag = nn.Linear(in_features, out_features)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.linear_real(z_real) - self.linear_imag(z_imag), \
               self.linear_real(z_imag) + self.linear_imag(z_real)

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, padding: int = 0) -> None:
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_real(z_real) - self.conv_imag(z_imag), \
               self.conv_real(z_imag) + self.conv_imag(z_real)