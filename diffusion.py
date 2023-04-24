import torch
import torch.functional as F
from torch import sqrt, nn
import numpy as np


class Diffusion(nn.Module):
    def __init__(self, model, steps=1000, length=88200) -> None:
        super(Diffusion, self).__init__()
        self.model = model

        self.steps = steps
        self.length = length

        start = 0.002
        end = 0.2
        a = 1.3

        t = torch.linspace(0, 1, steps, device=model.device)
        self.beta = start + (end - start) * 1 / \
            (1 + np.e ** -(a/(end - start)*(t - start-0.5)))
        self.beta = torch.linspace(start, end, steps, device=model.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        # create the noise and compute the noise audio-file
        noise = torch.randn_like(x)
        x_t = sqrt(self.alpha_hat[t]) * x + sqrt(1 - self.alpha_hat[t]) * noise

        # r = np.random.normal(1, 0.035)
        return x_t, noise

    @torch.no_grad()
    def sample(self, labels: list, loop=1):
        n = len(labels)
        model = self.model
        model.eval()

        # create a noise array that we want to denoise
        x = torch.randn(n, 1, self.length, device=model.device)
        l = torch.tensor(labels, device=model.device).view(-1, 1)

        print('Start creating Samples')

        # loop through all timesteps
        for i in range(1, self.steps):
            for _ in range(loop):
                # define the needed variables
                t = torch.ones(n, device=model.device).long() * \
                    (self.steps - i)
                alpha = self.alpha[t].view(-1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1)

                # predict the noise with the model
                timestamp = torch.ones(
                    n, device=model.device) * (self.steps - i)
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
