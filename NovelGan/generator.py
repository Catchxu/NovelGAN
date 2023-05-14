import torch
import torch.nn as nn
from .memory import MemoryUnit


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 2, 1)
            if down else
            nn.ConvTranspose1d(in_channels, out_channels, 3, 2, 1, 1),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, feature: int = 64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, feature, 7, 1, 3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_blocks = nn.Sequential(
            ConvBlock1d(feature, feature*2),
            ConvBlock1d(feature*2, feature*4),
            ConvBlock1d(feature*4, feature*4),
        )

        self.last = nn.Conv1d(feature*4, 1, 4, 1, 1)

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        return self.last(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int = 1, feature: int = 64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ConvTranspose1d(1, feature*4, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up_blocks = nn.Sequential(
            ConvBlock1d(feature*4, feature*4, down=False),
            ConvBlock1d(feature*4, feature*2, down=False),
            ConvBlock1d(feature*2, feature, down=False),
        )

        self.last = nn.ConvTranspose1d(feature, in_channels, 7, 1, 3)

    def forward(self, x):
        x = self.initial(x)
        x = self.up_blocks(x)
        return torch.tanh(self.last(x))


class Generator(nn.Module):
    '''
    This function is the generator in NovelGan, which includes two encoders, one
    decoder and one memory unit. All of them don't share the weight.

    Parameters
    ----------
    in_channels: int
        Number of input's channels.
    feature: int
        Dimension of latent feature space.
    mem_dim: int
        Number of the vectors in memory bank.
    z_dim: int
        Size of embedding vectors.

    Attributes
    ----------
    Encoder1: torch.nn.Module
        Encoder for getting the embedding vector (real_z) of real data (x).
    MemoryUnit: torch.nn.Module
        Memory bank class to reconstruct the real_z.
    Decoder: torch.nn.Module
        Decoder for getting the reconstruction data (fake_x) from mem_z
    Encoder2: torch.nn.Module
        Encoder for getting the embedding vector (fake_z) of fake_x

    Notes
    -----
    Forward process:
    x -> Encoder 1 -> MemoryUnit -> Decoder -> Encoder 2
                  |             |          |            |
                real_z        mem_z      fake_x       fake_z

    '''

    def __init__(self, in_channels: int, feature: int, mem_dim: int, z_dim: int):
        super().__init__()
        self.Encoder1 = Encoder(in_channels, feature)
        self.Memory = MemoryUnit(mem_dim, z_dim)
        self.Decoder = Decoder(in_channels, feature)
        self.Encoder2 = Encoder(in_channels, feature)

    def forward(self, x):
        '''
        Forward process of generator.

        Parameters
        ----------
        x: torch.Tensor
            The input data with shape of [batch_size, 1, n_var]

        Returns
        -------
        real_z: torch.Tensor
            The embedding vectors from x with shape of [batch_size, 1, z_dim]
        fake_x: torch.Tensor
            The reconstructed x from mem_z with shape of [batch_size, 1, n_var]
        fake_z: torch.Tensor
            The embedding vectors from fake_x with shape of [batch_size, 1, z_dim]

        '''

        real_z = self.Encoder1(x)
        mem_z = self.Memory(real_z)
        fake_x = self.Decoder(mem_z)
        fake_z = self.Encoder2(fake_x)

        return real_z, fake_x, fake_z


class Autoencoder(nn.Module):
    def __init__(self, in_channels: int, feature: int):
        super().__init__()
        self.Encoder = Encoder(in_channels, feature)
        self.Decoder = Decoder(in_channels, feature)

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x