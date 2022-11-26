import math
import torch
import torch.nn as nn
from typing import Tuple

def magnitude(z_real: torch.Tensor, z_imag: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(z_real ** 2 + z_imag ** 2 + 1e-14)

def complex_tanh(z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mag = torch.clamp(magnitude(z_real, z_imag), min=1e-8)
    tanh = torch.tanh(mag)
    return tanh * (z_real / mag), tanh * (z_imag / mag)

class ComplexLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(ComplexLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # i_t
        self.W_zi = nn.Parameter(torch.Tensor(input_size, hidden_size, 2))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size, 2))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # f_t
        self.W_zf = nn.Parameter(torch.Tensor(input_size, hidden_size, 2))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size, 2))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # c_t
        self.W_zc = nn.Parameter(torch.Tensor(input_size, hidden_size, 2))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size, 2))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size, 2))
        
        # o_t
        self.W_zo = nn.Parameter(torch.Tensor(input_size, hidden_size, 2))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size, 2))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, z: torch.Tensor) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_size, _, _ = z.size()
        hidden_seq = []
        h_t = torch.zeros(batch_size, self.hidden_size, 2).to(z.device)
        c_t = torch.zeros(batch_size, self.hidden_size, 2).to(z.device)
        for t in range(seq_size):
            z_t = z[:, t, ...]
            
            i_t = torch.sigmoid(z_t[...,0] @ self.W_zi[...,0] - z_t[...,1] @ self.W_zi[...,1] + \
                                h_t[...,0] @ self.W_hi[...,0] - h_t[...,1] @ self.W_hi[...,1] + \
                                self.b_i)
            
            f_t = torch.sigmoid(z_t[...,0] @ self.W_zf[...,0] - z_t[...,1] @ self.W_zf[...,1] + \
                                h_t[...,0] @ self.W_hf[...,0] - h_t[...,1] @ self.W_hf[...,1] + \
                                self.b_f)
            
            o_t = torch.sigmoid(z_t[...,0] @ self.W_zo[...,0] - z_t[...,1] @ self.W_zo[...,1] + \
                                h_t[...,0] @ self.W_ho[...,0] - h_t[...,1] @ self.W_ho[...,1] + \
                                self.b_o)

            c_tilde_t_real = z_t[...,0] @ self.W_zc[...,0] - z_t[...,1] @ self.W_zc[...,1] + \
                             h_t[...,0] @ self.W_hc[...,0] - h_t[...,1] @ self.W_hc[...,1] + \
                             self.b_c[...,0]
            c_tilde_t_imag = z_t[...,1] @ self.W_zc[...,0] + z_t[...,0] @ self.W_zc[...,1] + \
                             h_t[...,1] @ self.W_hc[...,0] + h_t[...,0] @ self.W_hc[...,1] + \
                             self.b_c[...,1]
            c_tilde_t_real, c_tilde_t_imag = complex_tanh(c_tilde_t_real, c_tilde_t_imag)

            c_t_real = f_t * c_t[...,0] + i_t * c_tilde_t_real
            c_t_imag = f_t * c_t[...,1] + i_t * c_tilde_t_imag

            c_t = torch.stack((c_t_real, c_t_imag), dim=-1)

            h_t_real, h_t_imag = complex_tanh(c_t_real, c_t_imag)
            h_t_real *= o_t
            h_t_imag *= o_t

            h_t = torch.stack((h_t_real, h_t_imag), dim=-1)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)