import torch
from torch import nn
import numpy as np
from torch import sin, cos, pow


class Sinusoidal(nn.Module):
    def __init__(self, device):
        super(Sinusoidal, self).__init__()
        self.device = device


def Sinu(length: int, n: int, position: int) -> torch.Tensor:
    values = torch.arange((n, 1, length)).float()
    output = torch.zeros_like((n, 1, length)).float()
    # print(length, n, position, values, output)
    print(values[::2].shape, output[::2].shape)
    print(sin(position / pow(1000, values[::2] / length)).shape, output.shape)
    output[::2] = sin(position / pow(1000, values[::2] / length))
    output[1::2] = cos(position / pow(1000, values[1::2] / length))

    return output


def dual(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv1d(in_channel + 2, out_channel, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channel, out_channel, kernel_size=9, padding=4),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


def up(in_channel, out_channel, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel + 2, in_channel, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channel, out_channel, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_channel, out_channel,
                           kernel_size=4, stride=4, output_padding=pad),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, device):
        super(UNet, self).__init__()
        self.device = device
        self.sinusoidal = Sinu

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

    def forward(self, x: torch.Tensor, timestamp: int, label: int) -> torch.Tensor:
        n = x.shape[0]
        label *= 100

        t1 = self.sinusoidal(88200, n, timestamp)
        t2 = self.sinusoidal(22050, n, timestamp)
        t3 = self.sinusoidal(5512, n, timestamp)
        t4 = self.sinusoidal(1378, n, timestamp)
        t5 = self.sinusoidal(344, n, timestamp)

        l1 = self.sinusoidal(88200, n, label)
        l2 = self.sinusoidal(22050, n, label)
        l3 = self.sinusoidal(5512, n, label)
        l4 = self.sinusoidal(1378, n, label)
        l5 = self.sinusoidal(344, n, label)

        x1 = self.down1(torch.cat([t1, l1, x], 1))
        x2 = self.down2(torch.cat([t2, l2, self.pool(x1)], 1))
        x3 = self.down3(torch.cat([t3, l3, self.pool(x2)], 1))
        x4 = self.down4(torch.cat([t4, l4, self.pool(x3)], 1))

        out = self.up4(torch.cat([t5, l5, self.pool(x4), 1]))
        out = self.up3(torch.cat([t4, l4, out, x4], 1))
        out = self.up2(torch.cat([t3, l3, out, x3], 1))
        out = self.up1(torch.cat([t2, l2, out, x2], 1))
        out = self.output(torch.cat([t1, l1, out, x1], 1))
        return out
