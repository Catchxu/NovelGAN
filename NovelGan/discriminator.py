import torch
import torch.nn as nn


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 4, 2, 1),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1, feature: int = 64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, feature, 7, 1, 3, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.down_blocks = nn.Sequential(
            ConvBlock1d(feature, feature*2),
            ConvBlock1d(feature*2, feature*4),
        )

        self.last = nn.Conv1d(feature*4, in_channels, 4, 1, 1)

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        return torch.sigmoid(self.last(x))