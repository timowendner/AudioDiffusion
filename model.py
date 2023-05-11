import torch
from torch import nn
import numpy as np
from torch import sin, cos, pow, Tensor
import torch.nn.functional as F


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
                              pow(1000, values[::2] / length))
        return output.view(n, 1, -1)


def dual(in_channel, out_channel, kernel=9):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.Dropout1d(p=0.2),
        nn.ReLU(),
        nn.Conv1d(out_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


def up(in_channel, out_channel, scale=2, kernel=9, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel, in_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.Dropout1d(p=0.2),
        nn.ReLU(),
        nn.Conv1d(in_channel, out_channel,
                  kernel_size=kernel, padding=kernel//2),
        nn.ReLU(),
        nn.ConvTranspose1d(out_channel, out_channel,
                           kernel_size=scale, stride=scale, output_padding=pad),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.device = config.device
        self.sinusoidal = Sinusoidal(config.device)

        # define the pooling layer
        length = config.audio_length
        scale = config.model_scale
        kernel = config.model_kernel
        self.pool = nn.MaxPool1d(kernel_size=scale, stride=scale)

        # define the encoder
        last = 1
        pad = []
        self.length = [length]
        self.down = nn.ModuleList([])
        for channel in config.model_layers:
            cur_pad, length = length % scale, length // scale
            self.length.append(length)
            pad.append(cur_pad)
            layer = dual(last+2, channel, kernel=kernel)
            self.down.append(layer)
            last = channel

        # define the decoder
        self.up = nn.ModuleList([])
        for channel in reversed(config.model_layers):
            layer = up(last+2, channel, scale=scale,
                       kernel=kernel, pad=pad.pop())
            self.up.append(layer)
            last = channel * 2

        # define the output layer
        output = nn.ModuleList([])
        last += 2
        for channel in config.model_out:
            conv = nn.Conv1d(
                last, channel, kernel_size=kernel, padding=kernel//2)
            output.append(conv)
            output.append(nn.ReLU())
            last = channel
        output.append(
            nn.Conv1d(last, 1, kernel_size=kernel, padding=kernel//2))
        self.output = nn.Sequential(*output)

    def forward(self, x: Tensor, timestamp: Tensor, label: Tensor) -> Tensor:
        timestamp = timestamp.to(self.device)
        label = (label * 100).to(self.device)

        # apply the encoder
        encoder = []
        length = [i for i in reversed(self.length)]
        for layer in self.down:
            t = self.sinusoidal(timestamp, length[-1])
            l = self.sinusoidal(label, length.pop())
            x = torch.cat([l, t, x], 1)
            x = layer(x)
            encoder.append(torch.cat([l, t, x], 1))
            x = self.pool(x)

        # apply the decoder
        t = self.sinusoidal(timestamp, length[-1])
        l = self.sinusoidal(label, length.pop())
        x = torch.cat([l, t, x], 1)
        for layer in self.up:
            x = layer(x)
            x = torch.cat([encoder.pop(), x], 1)

        # apply the output
        x = self.output(x)
        return x
