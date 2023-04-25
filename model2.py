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
        nn.Dropout(p=0.1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channel, out_channel, kernel_size=9, padding=4),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=True),
    )


def up(in_channel, out_channel, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel + 2, in_channel, kernel_size=9, padding=4),
        nn.Dropout(p=0.1),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channel, out_channel, kernel_size=9, padding=4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(out_channel, out_channel,
                           kernel_size=2, stride=2, output_padding=pad),
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
        self.step_count = config.step_count
        self.label_count = config.label_count

        c1, c2, c3, c4 = 8, 8, 12, 16
        c5, c6, c7, c8 = 24, 24, 32, 48

        self.down1 = dual(1, c1)
        self.down2 = dual(c1, c2)
        self.down3 = dual(c2, c3)
        self.down4 = dual(c3, c4)
        self.down5 = dual(c4, c5)
        self.down6 = dual(c5, c6)
        self.down7 = dual(c6, c7)
        self.down8 = dual(c7, c8)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.step_embedding = embedding(config.step_count, 344)
        self.label_embedding = embedding(config.label_count, 344)

        self.up8 = up(c8 + 2, c8, pad=1)  # 344
        self.up7 = up(c8*2, c7)  # 689
        self.up6 = up(c7*2, c6)  # 1378
        self.up5 = up(c6*2, c5)  # 2756
        self.up4 = up(c5*2, c4, pad=1)  # 5512
        self.up3 = up(c4*2, c3)  # 11050
        self.up2 = up(c3*2, c2)  # 22100
        self.up1 = up(c2*2, c1)  # 44200

        self.output = nn.Sequential(
            nn.Conv1d(c1*2 + 2, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=9, padding=4),
        )

    def forward(self, x: Tensor, timestamp: Tensor, label: Tensor) -> Tensor:
        timestamp = timestamp.to(self.device)
        label = label.to(self.device)
        n = x.shape[0]

        # create the sinusoidal
        t1 = self.sinusoidal(timestamp, 88200)
        t2 = self.sinusoidal(timestamp, 44100)
        t3 = self.sinusoidal(timestamp, 22050)
        t4 = self.sinusoidal(timestamp, 11025)
        t5 = self.sinusoidal(timestamp, 5512)
        t6 = self.sinusoidal(timestamp, 2756)
        t7 = self.sinusoidal(timestamp, 1378)
        t8 = self.sinusoidal(timestamp, 689)
        t9 = self.sinusoidal(timestamp, 344)

        l = label * 100
        l1 = self.sinusoidal(l, 88200)
        l2 = self.sinusoidal(l, 44100)
        l3 = self.sinusoidal(l, 22050)
        l4 = self.sinusoidal(l, 11025)
        l5 = self.sinusoidal(l, 5512)
        l6 = self.sinusoidal(l, 2756)
        l7 = self.sinusoidal(l, 1378)
        l8 = self.sinusoidal(l, 689)
        l9 = self.sinusoidal(l, 344)

        # create the embeddings
        timestamp = F.one_hot(timestamp.long(), self.step_count + 1)
        timestamp = timestamp.view(n, 1, -1).type(torch.float32)
        label = F.one_hot(label.long(), self.label_count + 1)
        label = label.view(n, 1, -1).type(torch.float32)
        step_emb = self.step_embedding(timestamp)
        label_emb = self.label_embedding(label)

        lengths = (88200, 22050, 11050, 5512, 2756, 1378, 689, 344)

        # encode the data into a lower dimensional space
        x1 = self.down1(torch.cat([t1, l1, x], 1))  # length 88200
        x2 = self.down2(torch.cat([t2, l2, self.pool(x1)], 1))  # length 44100
        x3 = self.down3(torch.cat([t3, l3, self.pool(x2)], 1))  # length 22050
        x4 = self.down4(torch.cat([t4, l4, self.pool(x3)], 1))  # length 11025
        x5 = self.down5(torch.cat([t5, l5, self.pool(x4)], 1))  # length 5512
        x6 = self.down6(torch.cat([t6, l6, self.pool(x5)], 1))  # length 2756
        x7 = self.down7(torch.cat([t7, l7, self.pool(x6)], 1))  # length 1378
        x8 = self.down8(torch.cat([t8, l8, self.pool(x7)], 1))  # length 689

        out = torch.cat([t9, l9, step_emb, label_emb, self.pool(x8)], 1)

        # decode the data
        out = self.up8(out)
        out = self.up7(torch.cat([t8, l8, x8, out], 1))
        out = self.up6(torch.cat([t7, l7, x7, out], 1))
        out = self.up5(torch.cat([t6, l6, x6, out], 1))
        out = self.up4(torch.cat([t5, l5, x5, out], 1))
        out = self.up3(torch.cat([t4, l4, x4, out], 1))
        out = self.up2(torch.cat([t3, l3, x3, out], 1))
        out = self.up1(torch.cat([t2, l2, x2, out], 1))
        out = self.output(torch.cat([t1, l1, x1, out], 1))
        return out
