import torch
import torch.functional as F
from torch import sqrt, nn


class Diffusion(nn.Module):
    def __init__(self, model, steps=1000, beta_start=1e-4, beta_end=0.02, length=88200) -> None:
        super(Diffusion, self).__init__()
        self.model = model

        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.length = length

        self.beta = torch.linspace(
            beta_start, beta_end, steps, device=model.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        # if the timestamp is 0 then return a noise-free version
        if t == 0:
            return x, torch.zeros_like(x)

        # create the noise and compute the noise audio-file
        noise = torch.randn_like(x)
        x_t = sqrt(self.alpha_hat[t]) * x + sqrt(1 - self.alpha_hat[t]) * noise

        # clamp the tensor and compute the noise distribution
        return x_t, noise

    @torch.no_grad()
    def sample(self, labels: list):
        n = len(labels)
        model = self.model
        model.eval()

        # create a noise array that we want to denoise
        x = torch.randn(n, 1, self.length, device=model.device)
        l = (torch.tensor(labels, device=model.device) * 100).view(-1, 1, 1)
        # l = torch.ones(n, device=model.device) * label

        # loop through all timesteps
        for i in range(1, self.steps):
            # define the needed variables
            t = torch.ones(n, device=model.device).long() * (self.steps - i)
            alpha = self.alpha[t].view(-1, 1, 1)
            alpha_hat = self.alpha_hat[t].view(-1, 1, 1)
            beta = self.beta[t].view(-1, 1, 1)

            # predict the noise with the model
            timestamp = (torch.ones_like(l, device=model.device)
                         * (self.steps - i)).view(-1, 1, 1)
            predicted_noise = model(x, timestamp, l)

            if i == self.steps - 1:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)

            x = 1 / sqrt(alpha) * (x - ((1 - alpha) / (sqrt(1 - alpha_hat)))
                                   * predicted_noise) + sqrt(beta) * noise
            # x = x.clamp(-1, 1)

            if i % 100 == 0:
                print(f'Step [{i}/{self.steps}]')

        # x = x.clamp(-1, 1)
        model.train()
        return x
