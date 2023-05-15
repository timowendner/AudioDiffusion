import torch
from torch import nn
import numpy as np
from torch import sin, cos, pow, Tensor
import torch.nn.functional as F


class Sinusoidal(nn.Module):
    def __init__(self, device):
        """
        Initializes a module that generates a tensor of sinusoidal functions

        Args:
            device (torch.device): The device on which the tensor is generated
        """
        super(Sinusoidal, self).__init__()
        self.device = device

    def forward(self, position: Tensor, length: int) -> Tensor:
        """
        Generates a tensor of sinusoidal functions with a given position and length

        Args:
            position (torch.Tensor): A tensor of shape (batch size, 1, 1) referring to timestamp or label
            length (int): The length of the sinusoidal function

        Returns:
            torch.Tensor: A tensor of shape (batch size, 1, length) containing the generated sinusoidal functions.
        """
        n = position.shape[0]
        values = torch.arange(length, device=self.device)
        output = torch.zeros((n, length), device=self.device)
        output[:, ::2] = sin(position.view(-1, 1) /
                             pow(1000, values[::2] / length))
        output[:, 1::2] = cos(position.view(-1, 1) /
                              pow(1000, values[::2] / length))
        return output.view(n, 1, -1)


def dual(in_channel: int, out_channel: int, kernel=9):
    """
    Constructs a dual convolutional block consisting of two 1D convolutional layers with dropout, batch normalization,
    and ReLU activation

    Args:
        in_channel (int): The number of input channels
        out_channel (int): The number of output channels
        kernel (int, optional): The size of the convolutional kernel. Defaults to 9

    Returns:
        nn.Sequential: The dual convolutional block.
    """
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
    """
    Constructs an upsampling block consisting of two 1D convolutional layers with dropout, batch normalization,
    ReLU activation, and a transposed convolutional layer

    Args:
        in_channel (int): The number of input channels
        out_channel (int): The number of output channels
        scale (int, optional): The scale factor for upsampling. Defaults to 2
        kernel (int, optional): The size of the convolutional kernel. Defaults to 9
        pad (int, optional): The amount of output padding. Defaults to 0

    Returns:
        nn.Sequential: The upsampling block
    """
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
        """
        Initializes the U-Net model for audio denoising

        Args:
            config (Config): the configuration object for the model
        """
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
        """
        Applies a forward pass through the U-Net model for audio denoising.

        Args:
            x (Tensor): the input audio signal, with shape (batch_size, 1, audio_length).
            timestamp (Tensor): the timestamp vector, with shape (batch_size, 1).
            label (Tensor): the label vector, with shape (batch_size, 1).

        Returns:
            Tensor: the denoised audio signal, with shape (batch_size, 1, audio_length).
        """
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
