import torch
from torch import nn


def dual(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channel, out_channel, kernel_size=9, padding=4),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


def up(in_channel, out_channel, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel, in_channel, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channel, out_channel, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_channel, out_channel,
                           kernel_size=4, stride=4, output_padding=pad),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


def sinusoidal(length: int,  label: int):
    pass


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = dual(1, 32)
        self.down2 = dual(32, 64)
        self.down3 = dual(64, 128)
        self.down4 = dual(128, 256)

        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.up4 = up(256, 256, pad=2)
        self.up3 = up(512, 128)
        self.up2 = up(256, 64, pad=2)
        self.up1 = up(128, 32)

        self.output = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.Conv1d(32, 32, kernel_size=9, padding=4),
            nn.Conv1d(32, 1, kernel_size=9, padding=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        out = self.up4(self.pool(x4))
        out = self.up3(torch.cat([out, x4], 1))
        out = self.up2(torch.cat([out, x3], 1))
        out = self.up1(torch.cat([out, x2], 1))

        out = self.output(torch.cat([out, x1], 1))
        return out
