import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, z_dim, shrink_thres=0.0025, bias=None):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.z_dim))  # MxC
        self.shrink_thres = shrink_thres
        self.bias = bias

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = input.reshape(input.shape[0], input.shape[2])  # input, B x 1 x C -> B x C

        att_weight = F.linear(input, self.weight)  # input x Mem^T, (BxC) x (CxM) = B x M
        att_weight = F.softmax(att_weight, dim=1)  # B x M
        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, M x C
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem, (BxM) x (MxC) = B x C

        output = output.reshape(output.shape[0], 1, -1)  # output,f B x C -> B x 1 x C
        return output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, z_dim={}'.format(
            self.mem_dim, self.z_dim is not None
        )


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output