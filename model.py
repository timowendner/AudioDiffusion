import torch
from torch import nn
import numpy as np
from torch import sin, cos, pow, Tensor


class Sinusoidal(nn.Module):
    def __init__(self, device):
        super(Sinusoidal, self).__init__()
        self.device = device

    def forward(self, length: int, position: torch.Tensor) -> torch.Tensor:
        n = position.shape[0]
        values = torch.arange(length, device=self.device).unsqueeze(
            0).unsqueeze(0).expand(n, 1, -1)
        output = torch.zeros_like(values, device=self.device)
        position = position.view(-1, 1, 1)

        output[:, :, ::2] = sin(
            position / pow(1000, values[:, :, ::2] / length))
        output[:, :, 1::2] = cos(
            position / pow(1000, values[:, :, 1::2] / length))
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


def embedding(in_channel, out_channel):
    return nn.Sequential(
        nn.Linear(1, in_channel),
        nn.ReLU(inplace=True),
        nn.Linear(in_channel, out_channel),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, device, step_count, label_count):
        super(UNet, self).__init__()
        self.device = device
        self.sinusoidal = Sinusoidal(device)

        self.down1 = dual(1, 48)
        self.down2 = dual(48, 96)
        self.down3 = dual(96, 192)
        self.down4 = dual(192, 384)

        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.step_embedding = embedding(step_count, 344)
        self.label_embedding = embedding(label_count, 344)

        self.up4 = up(384 + 2, 384, pad=2)
        self.up3 = up(768, 192)
        self.up2 = up(384, 96, pad=2)
        self.up1 = up(192, 48)

        self.output = nn.Sequential(
            nn.Conv1d(96 + 2, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=9, padding=4),
        )

    def forward(self, x: Tensor, timestamp: Tensor, label: Tensor) -> Tensor:

        t1 = self.sinusoidal(88200, timestamp)
        t2 = self.sinusoidal(22050, timestamp)
        t3 = self.sinusoidal(5512, timestamp)
        t4 = self.sinusoidal(1378, timestamp)
        t5 = self.sinusoidal(344, timestamp)

        l1 = self.sinusoidal(88200, label)
        l2 = self.sinusoidal(22050, label)
        l3 = self.sinusoidal(5512, label)
        l4 = self.sinusoidal(1378, label)
        l5 = self.sinusoidal(344, label)

        print(t1.shape, l1.shape, x.shape, timestamp.shape, label.shape)

        x1 = self.down1(torch.cat([t1, l1, x], 1))
        x2 = self.down2(torch.cat([t2, l2, self.pool(x1)], 1))
        x3 = self.down3(torch.cat([t3, l3, self.pool(x2)], 1))
        x4 = self.down4(torch.cat([t4, l4, self.pool(x3)], 1))

        step_embedding = self.step_embedding(timestamp)
        step_embedding = torch.unsqueeze(step_embedding, 1)
        label_embedding = self.label_embedding(label)
        label_embedding = torch.unsqueeze(label_embedding, 1)
        out = torch.cat(
            [t5, l5, step_embedding, label_embedding, self.pool(x4)], 1)

        out = self.up4(out)
        out = self.up3(torch.cat([t4, l4, out, x4], 1))
        out = self.up2(torch.cat([t3, l3, out, x3], 1))
        out = self.up1(torch.cat([t2, l2, out, x2], 1))
        out = self.output(torch.cat([t1, l1, out, x1], 1))
        return out
