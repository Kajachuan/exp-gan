import torch
import torch.nn as nn
from blocks import *
from lstm import ComplexLSTMLayer
from utils import *

class Model(nn.Module):
    def __init__(self, layers: int, bins: int) -> None:
        super(Model, self).__init__()
        self.layers = layers
        self.encoders = nn.ModuleList([EncoderBlock(2, 16)])
        self.encoders.extend([EncoderBlock(2**(4 + i), 2**(5 + i)) for i in range(layers - 3)])
        self.encoders.extend([EncoderBlock(2**(layers + 1), 2**(layers + 2), dropout=0.5)])
        self.encoders.extend([EncoderBlock(2**(layers + 2), 2**(layers + 3), dropout=0.5, pooling=False)])

        self.out_enc_size = bins // (2 ** (layers - 1))

        self.lstm = ComplexLSTMLayer(input_size=self.out_enc_size * (2 ** (self.layers + 3)), hidden_size=512)
        self.linear = ComplexLinear(in_features=512, out_features=self.out_enc_size * (2 ** (self.layers + 3)))

        self.decoders = nn.ModuleList([DecoderBlock(2**i, 2**(i-1), 2**(i-1)) 
                                       for i in range(layers + 3, 4, -1)])

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
            sizes.append(z_real.size())
            ((z_real, z_imag), idx), skip = self.encoders[i](z_real, z_imag)
            indices.append(idx)
            residual.append(skip)

        z_real, z_imag = self.encoders[-1](z_real, z_imag)

        z_real = z_real.transpose(1, 3).reshape(batch_size, -1, self.out_enc_size * (2 ** (self.layers + 3)))
        z_imag = z_imag.transpose(1, 3).reshape(batch_size, -1, self.out_enc_size * (2 ** (self.layers + 3)))

        (z_real, z_imag), _ = self.lstm(z_real, z_imag)
        z_real, z_imag = self.linear(z_real, z_imag)

        z_real = z_real.reshape(batch_size, -1, self.out_enc_size, (2 ** (self.layers + 3))).transpose(1, 3)
        z_imag = z_imag.reshape(batch_size, -1, self.out_enc_size, (2 ** (self.layers + 3))).transpose(1, 3)

        for i in range(self.layers - 1):
            indice_real, indice_imag = indices.pop()
            skip_real, skip_imag = residual.pop()
            output_size = sizes.pop()
            z_real, z_imag = self.decoders[i](z_real, z_imag,
                                              torch.repeat_interleave(indice_real, 2, dim=1), 
                                              torch.repeat_interleave(indice_imag, 2, dim=1),
                                              skip_real, skip_imag,
                                              output_size)
        z_real, z_imag = self.conv(z_real, z_imag)
        z_real, z_imag = self.relu(z_real, z_imag)
        z_real, z_imag = self.out_conv(z_real, z_imag)
        z_real, z_imag = self.sigmoid(z_real, z_imag)
        z_real = z_real.reshape(z_real.size(0), 4, 2, z_real.size(2), z_real.size(3)).transpose(0,1)
        z_imag = z_imag.reshape(z_imag.size(0), 4, 2, z_imag.size(2), z_imag.size(3)).transpose(0,1)
        out_real = z_real * orig_real - z_imag * orig_imag
        out_imag = z_real * orig_imag + z_imag * orig_real
        return out_real.transpose(0,1), out_imag.transpose(0,1)