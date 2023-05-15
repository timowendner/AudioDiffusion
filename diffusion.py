import torch
import torch.functional as F
from torch import sqrt, nn
import numpy as np
from itertools import chain


class Diffusion(nn.Module):
    def __init__(self, config) -> None:
        """
        Initializes a diffusion model

        Args:
            config (Config): the configuration object for the model

        Raises:
            AttributeError: If an unknown beta schedule is provided in the configuration
        """
        super(Diffusion, self).__init__()
        self.steps = config.step_count
        self.length = config.audio_length

        start = config.beta_start
        end = config.beta_end

        # create the beta schedule
        t = torch.linspace(0, 1, self.steps, device=config.device)
        if config.beta_schedule == 'linear':
            self.beta = start + (end - start) * t
        elif config.beta_schedule == 'quadratic':
            self.beta = start + (end - start) * t**2
        else:
            raise AttributeError(
                f'Beta Schedule {config.beta_schedule} is unknown')

        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        """adds noise to an audio-file

        Args:
            x (torch.tensor): The input audio-file
            t (int): The timestamp for the noise generation

        Returns:
            x_t (torch.Tensor): The audio-file with added noise.
            noise (torch.Tensor): The noise that was added to the audio-file.
        """
        noise = torch.randn_like(x)
        x_t = sqrt(self.alpha_hat[t]) * x + sqrt(1 - self.alpha_hat[t]) * noise

        return x_t, noise

    @torch.no_grad()
    def sample(self, model, config):
        """
        Generates new audio samples using a specified model

        Args:
            model (torch.nn.Module): The model used to generate the samples
            config (Config): the configuration object for the model

        Returns:
            torch.Tensor: A tensor containing the generated audio samples. The shape of the tensor is
                (samples, 1, audio_length), where 'samples' is the number of generated audio samples, '1'
                indicates a single audio channel (mono), and 'audio_length' is the duration of each audio sample
        """
        n = 1
        labels = [config.create_label + 1] * n
        model.eval()

        # create a noise array that we want to denoise
        x = torch.randn(n, 1, self.length, device=config.device)
        l = torch.tensor(labels, device=config.device).view(-1, 1)

        # loop through all timesteps
        for i in chain(range(1, self.steps), [self.steps - 1]*config.create_last):
            for _ in range(config.create_loop):
                # define the needed variables
                t = torch.ones(n, device=config.device).long() * \
                    (self.steps - i)
                alpha = self.alpha[t].view(-1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1)

                # predict the noise with the model
                timestamp = torch.ones(
                    n, device=config.device) * (self.steps - i)
                timestamp = timestamp.view(-1, 1)
                predicted_noise = model(x, timestamp, l)

                # remove the predicted noise and add new noise
                if i == self.steps - 1:
                    noise = torch.zeros_like(x)
                else:
                    noise = torch.randn_like(x)
                x = 1 / sqrt(alpha) * (x - ((1 - alpha) / (sqrt(1 - alpha_hat)))
                                       * predicted_noise) + sqrt(beta) * noise

        model.train()
        return x
