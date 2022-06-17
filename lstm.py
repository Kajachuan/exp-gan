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
        self.W_zi_real = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_zi_imag = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi_real = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hi_imag = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i_real = nn.Parameter(torch.Tensor(hidden_size))
        self.b_i_imag = nn.Parameter(torch.Tensor(hidden_size))
        
        # f_t
        self.W_zf_real = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_zf_imag = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf_real = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hf_imag = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f_real = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f_imag = nn.Parameter(torch.Tensor(hidden_size))

        # c_t
        self.W_zc_real = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_zc_imag = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hc_real = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_hc_imag = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c_real = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c_imag = nn.Parameter(torch.Tensor(hidden_size))
        
        # o_t
        self.W_zo_real = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_zo_imag = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho_real = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ho_imag = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o_real = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o_imag = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], 
            Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_size, _ = z_real.size()
        hidden_seq_real, hidden_seq_imag = [], []
        h_t_real = torch.zeros(batch_size, self.hidden_size).to(z_real.device)
        h_t_imag = torch.zeros(batch_size, self.hidden_size).to(z_imag.device)
        c_t_real = torch.zeros(batch_size, self.hidden_size).to(z_real.device)
        c_t_imag = torch.zeros(batch_size, self.hidden_size).to(z_imag.device)
        for t in range(seq_size):
            z_t_real = z_real[:, t, :]
            z_t_imag = z_imag[:, t, :]
            
            i_t = torch.sigmoid(z_t_real @ self.W_zi_real - z_t_imag @ self.W_zi_imag + \
                                h_t_real @ self.W_hi_real - h_t_imag @ self.W_hi_imag + \
                                self.b_i_real)
            
            f_t = torch.sigmoid(z_t_real @ self.W_zf_real - z_t_imag @ self.W_zf_imag + \
                                h_t_real @ self.W_hf_real - h_t_imag @ self.W_hf_imag + \
                                self.b_f_real)
            
            o_t = torch.sigmoid(z_t_real @ self.W_zo_real - z_t_imag @ self.W_zo_imag + \
                                h_t_real @ self.W_ho_real - h_t_imag @ self.W_ho_imag + \
                                self.b_o_real)

            c_tilde_t_real = z_t_real @ self.W_zc_real - z_t_imag @ self.W_zc_imag + \
                             h_t_real @ self.W_hc_real - h_t_imag @ self.W_hc_imag + \
                             self.b_c_real
            c_tilde_t_imag = z_t_imag @ self.W_zc_real + z_t_real @ self.W_zc_imag + \
                             h_t_imag @ self.W_hc_real + h_t_real @ self.W_hc_imag + \
                             self.b_c_imag
            c_tilde_t_real, c_tilde_t_imag = complex_tanh(c_tilde_t_real, c_tilde_t_imag)

            c_t_real = f_t * c_t_real + i_t * c_tilde_t_real
            c_t_imag = f_t * c_t_imag + i_t * c_tilde_t_imag

            h_t_real, h_t_imag = complex_tanh(c_t_real, c_t_imag)
            h_t_real *= o_t
            h_t_imag *= o_t

            hidden_seq_real.append(h_t_real.unsqueeze(0))
            hidden_seq_imag.append(h_t_imag.unsqueeze(0))

        hidden_seq_real = torch.cat(hidden_seq_real, dim=0)
        hidden_seq_real = hidden_seq_real.transpose(0, 1).contiguous()
        hidden_seq_imag = torch.cat(hidden_seq_imag, dim=0)
        hidden_seq_imag = hidden_seq_imag.transpose(0, 1).contiguous()

        return (hidden_seq_real, hidden_seq_imag), ((h_t_real, h_t_imag), (c_t_real, c_t_imag))