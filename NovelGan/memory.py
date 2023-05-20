import torch
from torch import nn
import math
from torch.nn import functional as F


class MemoryUnit(nn.Module):
    '''
    This class includes memory bank construction based on Soft Attention.

    Parameters
    ----------
    mem_dim: int
        Number of the vectors in memory bank.
    z_dim: int
        Size of embedding vectors.
    shrink_thres: float
        Threshold in hard_shrink_relu.

    Attributes
    ----------
    mem_dim: int
        Number of the vectors in memory bank.
    z_dim: int
        Size of embedding vectors.
    shrink_thres: float
        Threshold in hard_shrink_relu.
    mem: torch.Tensor
        Memory bank for reconstruction, with shape of [mem_dim, z_dim].
    mem_ptr: torch.Tensor
        Pointer of the memory bank's head.

    '''

    def __init__(self, mem_dim, z_dim, shrink_thres=0.0035):
        super().__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.shrink_thres = shrink_thres
        self.register_buffer("mem", torch.randn(self.mem_dim, self.z_dim))
        self.register_buffer("mem_ptr", torch.zeros(1, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def update_mem(self, z):
        z = z.reshape(z.shape[0], z.shape[2])
        batch_size = z.shape[0]  # z, B x C
        ptr = int(self.mem_ptr)
        assert self.mem_dim % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.mem[ptr:ptr + batch_size, :] = z  # mem, M x C
        ptr = (ptr + batch_size) % self.mem_dim  # move pointer

        self.mem_ptr[0] = ptr

    def attention(self, input):
        '''
        Calculate the attention weight by dot product attention.
        '''

        att_weight = torch.mm(input, self.mem.T)  # input x mem^T, (BxC) x (CxM) = B x M
        att_weight = F.softmax(att_weight, dim=1)  # B x M

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)  # AttWeight x mem, (BxM) x (MxC) = B x C

        return output

    def forward(self, input):
        input = input.reshape(input.shape[0], input.shape[2])  # input, B x 1 x C -> B x C
        output = self.attention(input)
        output = output.reshape(output.shape[0], 1, -1)  # output,f B x C -> B x 1 x C

        return output

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output