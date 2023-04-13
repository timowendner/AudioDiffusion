import torch


class Diffusion:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02, length=88200) -> None:
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.length = length

        self.beta = torch.linspace(beta_start, beta_end, steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise(self, x, t):
        if t == 0:
            return x
        x_t = torch.sqrt(self.alpha_hat[t]) * x \
            + torch.sqrt(1 - self.alpha_hat[t]) * torch.randn_like(x)
        x_t = x_t.clamp(-1, 1)
        return x_t

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.length)).to(model.device)
            for i in reversed(range(1, self.steps)):
                t = (torch.ones(n) * i).long().to(model.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) \
                    * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                       * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x
