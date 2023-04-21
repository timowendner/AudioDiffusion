import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import sin, cos, pow, Tensor


class Sinusoidal(nn.Module):
    def __init__(self, device):
        super(Sinusoidal, self).__init__()
        self.device = device

    def forward(self, position: torch.Tensor, length: int) -> torch.Tensor:
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
        nn.Conv1d(in_channel + 2, out_channel, kernel_size=9, padding=4),
        # nn.Dropout(p=0.1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channel, out_channel, kernel_size=9, padding=4),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


def up(in_channel, out_channel, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel + 2, in_channel, kernel_size=9, padding=4),
        # nn.Dropout(p=0.1),
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
        nn.Linear(in_channel, out_channel),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, device, step_count, label_count):
        super(UNet, self).__init__()
        self.device = device
        self.sinusoidal = Sinusoidal(device)
        self.step_count = step_count
        self.label_count = label_count

        self.down1 = dual(1, 16)
        self.down2 = dual(16, 32)
        self.down3 = dual(32, 64)
        self.down4 = dual(64, 128)

        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.step_embedding = embedding(step_count, 344)
        self.label_embedding = embedding(label_count, 344)

        self.up4 = up(128 + 2, 128, pad=2)
        self.up3 = up(256, 64)
        self.up2 = up(128, 32, pad=2)
        self.up1 = up(32, 16)

        self.output = nn.Sequential(
            nn.Conv1d(96 + 2, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=9, padding=4),
        )
        self.test = nn.Conv1d(3, 3, kernel_size=9, padding=4)

    def forward(self, x: Tensor, timestamp: Tensor, label: Tensor) -> Tensor:
        timestamp = timestamp.to(self.device)
        label = label.to(self.device)
        n = x.shape[0]

        t1 = self.sinusoidal(timestamp, 88200)
        t2 = self.sinusoidal(timestamp, 22050)
        t3 = self.sinusoidal(timestamp, 5512)
        t4 = self.sinusoidal(timestamp, 1378)
        t5 = self.sinusoidal(timestamp, 344)

        l1 = self.sinusoidal(label, 88200)
        l2 = self.sinusoidal(label, 22050)
        l3 = self.sinusoidal(label, 5512)
        l4 = self.sinusoidal(label, 1378)
        l5 = self.sinusoidal(label, 344)

        timestamp = F.one_hot(
            timestamp.long(), self.step_count + 1).view(n, 1, -1)
        label = F.one_hot(label.long(), self.step_count + 1).view(n, 1, -1)

        print(t1.shape, l1.shape, x.shape, label.shape, timestamp.shape)
        print(t1.device, l1.device, x.device, label.device, timestamp.device)
        x = torch.cat([t1, l1, x], 1)
        print(x.shape, x.dtype)
        x = self.test(x)
        print(x.shape, x.dtype)
        x1 = self.down1(x)
        x2 = self.down2(torch.cat([t2, l2, self.pool(x1)], 1))
        x3 = self.down3(torch.cat([t3, l3, self.pool(x2)], 1))
        x4 = self.down4(torch.cat([t4, l4, self.pool(x3)], 1))

        step_embedding = self.step_embedding(timestamp)
        label_embedding = self.label_embedding(label)
        print(step_embedding.shape, label_embedding.shape)
        out = torch.cat(
            [t5, l5, step_embedding, label_embedding, self.pool(x4)], 1)

        print(out.shape)

        out = self.up4(out)
        out = self.up3(torch.cat([t4, l4, out, x4], 1))

        out = self.up2(torch.cat([t3, l3, out, x3], 1))
        print(t2.shape, l2.shape, out.shape,
              x.shape. label.shape, timestamp.shape)
        out = self.up1(torch.cat([t2, l2, out, x2], 1))
        out = self.output(torch.cat([t1, l1, out, x1], 1))
        return out
