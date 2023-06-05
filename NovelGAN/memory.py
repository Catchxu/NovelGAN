import torch
from torch import nn
import math
from torch.nn import functional as F
from ._utils import hard_shrink_relu


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, z_dim, shrink_thres=0.0025):
        super().__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.shrink_thres = shrink_thres
        self.mem = nn.Parameter(torch.randn(self.mem_dim, self.z_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    def attention(self, input):
        # norm_input = F.normalize(input, p=2, dim=1)
        # norm_mem = F.normalize(self.mem, p=2, dim=1)
        # att_weight = torch.mm(norm_input, norm_mem.T)  # input x mem^T, (BxC) x (CxM) = B x M
        att_weight = torch.mm(input, self.mem.T)
        att_weight = F.softmax(att_weight, dim=1)  # B x M

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)  # AttWeight x mem, (BxM) x (MxC) = B x C

        return output

    def forward(self, x):
        x = self.attention(x)
        return x