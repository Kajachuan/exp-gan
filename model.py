from turtle import forward
import torch
import torch.nn as nn
from blocks import *
from lstm import ComplexLSTMLayer
from utils import *

class Model(nn.Module):
    def __init__(self, layers: int, bins: int) -> None:
        super(Model, self).__init__()
        self.layers = layers
        self.encoders = nn.ModuleList([EncoderBlock(2, 32)])
        self.encoders.extend([EncoderBlock(2**(5 + i), 2**(6 + i)) for i in range(layers - 3)])
        self.encoders.extend([EncoderBlock(2**(layers + 2), 2**(layers + 3), dropout=0.5)])
        self.encoders.extend([EncoderBlock(2**(layers + 3), 2**(layers + 4), dropout=0.5, pooling=False)])

        self.out_enc_size = bins // (layers - 1)

        self.lstm = ComplexLSTMLayer(input_size=self.out_enc_size, hidden_size=1024)
        self.linear = ComplexLinear(in_features=1024, out_features=self.out_enc_size)

        self.decoders = nn.ModuleList([DecoderBlock(2**i, 2**(i-1), 2**(i-1)) 
                                       for i in range(layers + 4, 4, -1)])

        self.conv = ComplexConv2d(32, 16, 3, 1)
        self.relu = ComplexReLU()
        self.out_conv = ComplexConv2d(16, 8, 1)
        self.sigmoid = ComplexSigmoid()

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor):
        indices = []
        residual = []
        sizes = []

        orig_real = torch.clone(z_real)
        orig_imag = torch.clone(z_imag)

        batch_size = z_real.size(0)

        for i in range(self.layers - 1):
            ((z_real, z_imag), idx), skip = self.encoders[i](z_real, z_imag)
            indices.append(idx)
            residual.append(skip)
            sizes.append(skip[0].size())

        z_real, z_imag = self.encoders[self.layers - 1](z_real, z_imag)

        z_real = z_real.transpose(1, 3).reshape(batch_size, -1, self.out_enc_size * (2 ** (self.layers + 4)))
        z_imag = z_imag.transpose(1, 3).reshape(batch_size, -1, self.out_enc_size * (2 ** (self.layers + 4)))

        z_real, z_imag = self.lstm(z_real, z_imag)
        z_real, z_imag = self.linear(z_real, z_imag)

        z_real = z_real.reshape(batch_size, -1, self.out_enc_size, (2 ** (self.layers + 4))).transpose(1, 3)
        z_imag = z_imag.reshape(batch_size, -1, self.out_enc_size, (2 ** (self.layers + 4))).transpose(1, 3)

        for i in range(self.layers - 1):
            indice_real, indice_imag = indices.pop()
            skip_real, skip_imag = residual.pop()
            output_size = sizes.pop()
            z_real, z_imag = self.decoders[i](z_real, z_imag,
                                              indice_real, indice_imag,
                                              skip_real, skip_imag,
                                              output_size)
        z_real, z_imag = self.conv(z_real, z_imag)
        z_real, z_imag = self.relu(z_real, z_imag)
        z_real, z_imag = self.out_conv(z_real, z_imag)
        z_real, z_imag = self.sigmoid(z_real, z_imag)
        return z_real * orig_real - z_imag * orig_imag, \
               z_real * orig_imag + z_imag * orig_real