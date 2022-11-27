import torch
import torch.nn as nn
from blocks import *
from lstm import ComplexBLSTMLayer
from utils import *

class Model(nn.Module):
    def __init__(self, layers: int, nfft: int) -> None:
        super(Model, self).__init__()

        bins = nfft // 2 + 1

        self.layers = layers
        self.encoders = nn.ModuleList([EncoderBlock(2, 16)])
        self.encoders.extend([EncoderBlock(2**(4 + i), 2**(5 + i)) for i in range(layers - 3)])
        self.encoders.extend([EncoderBlock(2**(layers + 1), 2**(layers + 2), dropout=0.5)])
        self.encoders.extend([EncoderBlock(2**(layers + 2), 2**(layers + 3), dropout=0.5, pooling=False)])

        self.out_enc_size = bins // (2 ** (layers - 1))
        self.features_num = self.out_enc_size * (2 ** (self.layers + 3))

        self.blstm = ComplexBLSTMLayer(input_size=self.features_num, hidden_size=256)
        self.linear = ComplexLinear(in_features=512, out_features=self.features_num)

        self.decoders = nn.ModuleList([DecoderBlock(2**i, 2**(i-1), 2**(i-1)) 
                                       for i in range(layers + 3, 4, -1)])

        self.conv = ComplexConv2d(16, 8, 3, 1)
        self.relu = nn.ReLU()
        self.out_conv = ComplexConv2d(8, 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor):
        indices = []
        residual = []
        sizes = []

        orig = torch.clone(z)

        batch_size = z.size(0)

        for i in range(self.layers - 1):
            sizes.append(z[...,0].size())
            (z, idx), skip = self.encoders[i](z)
            indices.append(idx)
            residual.append(skip)

        z = self.encoders[-1](z)

        z = z.transpose(1, 3).reshape(batch_size, -1, self.features_num, 2)

        z = self.blstm(z)
        z = self.linear(z)

        z = z.reshape(batch_size, -1, self.out_enc_size, (2 ** (self.layers + 3)), 2).transpose(1, 3)

        for i in range(self.layers - 1):
            idx = indices.pop()
            skip = residual.pop()
            output_size = sizes.pop()
            z = self.decoders[i](z, torch.repeat_interleave(idx, 2, dim=1), skip, output_size)
        z = self.conv(z)
        z = self.relu(z)
        z = self.out_conv(z)
        z = self.sigmoid(z)
        z = z.reshape(z.size(0), 4, 2, z.size(2), z.size(3), 2).transpose(0,1)
        out_real = z[...,0] * orig[...,0] - z[...,1] * orig[...,1]
        out_imag = z[...,0] * orig[...,1] + z[...,1] * orig[...,0]
        return torch.stack((out_real, out_imag), dim=-1).transpose(0,1)