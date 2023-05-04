import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import sin, cos, pow, Tensor


class Sinusoidal(nn.Module):
    def __init__(self, device):
        super(Sinusoidal, self).__init__()
        self.device = device

    def forward(self, position: Tensor, length: int) -> Tensor:
        n = position.shape[0]
        values = torch.arange(length, device=self.device)
        output = torch.zeros((n, length), device=self.device)
        output[:, ::2] = sin(position.view(-1, 1) /
                             pow(1000, values[::2] / length))
        output[:, 1::2] = cos(position.view(-1, 1) /
                              pow(1000, values[1::2] / length))
        return output.view(n, 1, -1)


def dual(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv1d(in_channel + 2, out_channel, kernel_size=3, padding=1),
        nn.Dropout(p=0.1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channel, out_channel, kernel_size=5, padding=2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channel, out_channel, kernel_size=7, padding=3),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


def up(in_channel, out_channel, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel + 2, in_channel, kernel_size=7, padding=3),
        nn.Dropout(p=0.1),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channel, out_channel, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_channel, out_channel,
                           kernel_size=4, stride=4, output_padding=pad),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


def embedding(in_channel, out_channel):
    return nn.Sequential(
        nn.Linear(in_channel+1, out_channel),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, device, config):
        super(UNet, self).__init__()
        self.device = device
        self.sinusoidal = Sinusoidal(device)
        # for hp search
        self.step_count = config['step_count']
        self.label_count = config['label_count']
        # self.step_count = config.step_count
        # self.label_count = config.label_count
        # self.name = 'UNet4'

        c1, c2, c3, c4 = 24, 32, 64, 96

        self.down1 = dual(1, c1)
        self.down2 = dual(c1, c2)
        self.down3 = dual(c2, c3)
        self.down4 = dual(c3, c4)

        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.step_embedding = embedding(self.step_count, 344)
        self.label_embedding = embedding(self.label_count, 344)

        self.up4 = up(c4 + 2, c4, pad=2)
        self.up3 = up(c4*2, c3)
        self.up2 = up(c3*2, c2, pad=2)
        self.up1 = up(c2*2, c1)

        self.output = nn.Sequential(
            nn.Conv1d(c1*2 + 2, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, x: Tensor, timestamp: Tensor, label: Tensor) -> Tensor:
        timestamp = timestamp.to(self.device)
        label = label.to(self.device)
        n = x.shape[0]

        t1 = self.sinusoidal(timestamp, 88200)
        t2 = self.sinusoidal(timestamp, 22050)
        t3 = self.sinusoidal(timestamp, 5512)
        t4 = self.sinusoidal(timestamp, 1378)
        t5 = self.sinusoidal(timestamp, 344)

        l = label * 100
        l1 = self.sinusoidal(l, 88200)
        l2 = self.sinusoidal(l, 22050)
        l3 = self.sinusoidal(l, 5512)
        l4 = self.sinusoidal(l, 1378)
        l5 = self.sinusoidal(l, 344)

        timestamp = F.one_hot(timestamp.long(), self.step_count + 1)
        timestamp = timestamp.view(n, 1, -1).type(torch.float32)
        label = F.one_hot(label.long(), self.label_count + 1)
        label = label.view(n, 1, -1).type(torch.float32)
        x = torch.cat([t1, l1, x], 1)
        x1 = self.down1(x)
        x2 = self.down2(torch.cat([t2, l2, self.pool(x1)], 1))
        x3 = self.down3(torch.cat([t3, l3, self.pool(x2)], 1))
        x4 = self.down4(torch.cat([t4, l4, self.pool(x3)], 1))

        step_emb = self.step_embedding(timestamp)
        label_emb = self.label_embedding(label)
        out = torch.cat([t5, l5, step_emb, label_emb, self.pool(x4)], 1)

        out = self.up4(out)
        out = self.up3(torch.cat([t4, l4, out, x4], 1))
        out = self.up2(torch.cat([t3, l3, out, x3], 1))
        out = self.up1(torch.cat([t2, l2, out, x2], 1))
        out = self.output(torch.cat([t1, l1, out, x1], 1))
        return out
