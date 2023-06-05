import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory import MemoryUnit


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm: bool = True, act: bool = True, use_dropout=False):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear(x)
        return self.dropout(x) if self.use_dropout else x


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256]):
        super().__init__()
        self.down = nn.Sequential(
            LinearBlock(in_dim, out_dim[0]),
            LinearBlock(out_dim[0], out_dim[1]),
            LinearBlock(out_dim[1], out_dim[2], act=False),
        )
        self.bottleneck = LinearBlock(out_dim[2], out_dim[2], use_dropout=False)

    def forward(self, x):
        x = self.down(x)
        return self.bottleneck(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256]):
        super().__init__()
        self.up = nn.Sequential(
            LinearBlock(out_dim[2], out_dim[1], use_dropout=False),
            LinearBlock(out_dim[1], out_dim[0], use_dropout=False),
            LinearBlock(out_dim[0], in_dim, norm=False, act=False),
        )

    def forward(self, x):
        x = self.up(x)
        return torch.tanh(x)


class Memory_G(nn.Module):
    def __init__(self, in_dim, out_dim=[1024, 512, 256], mem_dim=2048):
        super().__init__()
        self.Encoder1 = Encoder(in_dim, out_dim)
        self.Memory = MemoryUnit(mem_dim, out_dim[2])
        self.Decoder = Decoder(in_dim, out_dim)
        self.Encoder2 = Encoder(in_dim, out_dim)

    def forward(self, x):
        real_z = self.Encoder1(x)
        mem_z = self.Memory(real_z)
        fake_x = self.Decoder(mem_z)
        fake_z = self.Encoder2(fake_x)
        return real_z, fake_x, fake_z


class Align_G(nn.Module):
    def __init__(self, base_cells, input_cells, in_dim, out_dim=[1024, 512, 256]):
        super().__init__()
        self.Encoder = Encoder(in_dim, out_dim)
        self.Match = nn.Parameter(torch.Tensor(base_cells, input_cells))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Match.size(1))
        self.Match.data.uniform_(-stdv, stdv)

    def forward(self, x, base):
        x = self.Encoder(x)
        fake_base = torch.mm(F.relu(self.Match), x)
        base = self.Encoder(base)
        return fake_base, base, self.Match
