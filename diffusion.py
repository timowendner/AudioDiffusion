import torch
import torch.functional as F
from torch import sqrt, nn
import numpy as np


class Diffusion(nn.Module):
    def __init__(self, config) -> None:
        super(Diffusion, self).__init__()

        self.steps = config.step_count
        self.length = config.audio_length

        start = config.beta_start
        end = config.beta_end
        a = config.beta_sigmoid

        t = torch.linspace(0, 1, self.steps, device=config.device)
        self.beta = start + (end - start) * 1 / \
            (1 + np.e ** -(a/(end - start)*(t - start-0.5)))
        self.beta = start + (end - start) * t**2
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        # create the noise and compute the noise audio-file
        noise = torch.randn_like(x)
        x_t = sqrt(self.alpha_hat[t]) * x + sqrt(1 - self.alpha_hat[t]) * noise

        # r = np.random.normal(1, 0.035)
        return x_t, noise

    @torch.no_grad()
    def sample(self, model, config):
        n = config.create_count
        labels = [config.create_label + 1] * n
        model.eval()

        # create a noise array that we want to denoise
        x = torch.randn(n, 1, self.length, device=config.device)
        l = torch.tensor(labels, device=config.device).view(-1, 1)

        print('Start creating Samples')

        # loop through all timesteps
        for i in range(1, self.steps):
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

                if i == self.steps - 1:
                    noise = torch.zeros_like(x)
                else:
                    noise = torch.randn_like(x)

                x = 1 / sqrt(alpha) * (x - ((1 - alpha) / (sqrt(1 - alpha_hat)))
                                       * predicted_noise) + sqrt(beta) * noise

            if i % (self.steps // 10) == 0:
                print(f'Step [{i}/{self.steps}]')

        model.train()
        return x
