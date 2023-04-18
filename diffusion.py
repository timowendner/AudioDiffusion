import torch
import torch.functional as F
from torch import sqrt, nn


class Diffusion(nn.Module):
    def __init__(self, model, steps=1000, beta_start=1e-4, beta_end=0.02, length=88200) -> None:
        super(Diffusion, self).__init__()
        self.model = model
        # self.to(model.device)

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
        x_t = x_t.clamp(-1, 1)
        return x_t, x_t - x

    @torch.no_grad()
    def sample(self, n: int, label: int):
        model = self.model
        model.eval()

        # create a noise array that we want to denoise
        x = torch.randn(n, 1, self.length, device=model.device)
        # l = torch.ones(n, device=model.device) * label

        # loop through all timesteps

        for i in reversed(range(1, self.steps)):
            t = (torch.ones(n) * i).long().to(model.device)
            alpha = self.alpha[t].unsqueeze(0).unsqueeze(0)
            alpha_hat = self.alpha_hat[t].unsqueeze(0).unsqueeze(0)
            beta = self.beta[t].unsqueeze(0).unsqueeze(0)
            predicted_noise = model(x, t, label)
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            print(alpha.shape, beta.shape, t.shape, x.shape)
            x = 1 / sqrt(alpha) * (x - ((1 - alpha) / (sqrt(1 - alpha_hat)))
                                   * predicted_noise) + sqrt(beta) * noise
            x = x.clamp(-1, 1)
        # model.train()
        return x

    # def loss(self, x0: torch.Tensor, noise: torch.Tensor = None):
    #     batch_size = x0.shape[0]
    #     t = torch.randint(0, self.n_steps, (batch_size,),
    #                       device=x0.device)
    #     if noise is None:
    #         noise = torch.randn_like(x0)
    #     xt = self.q_sample(x0, t, eps=noise)
    #     eps_theta = self.eps_model(xt, t)
    #     return F.mse_loss(noise, eps_theta)
