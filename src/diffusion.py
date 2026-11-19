"""Forward + Reverse Diffusion - PDF wala idea."""

import torch
import numpy as np

class Diffusion:
    def __init__(self, n_steps=400, device="cpu"):
        self.n_steps = n_steps
        self.device = device
        # beta - har step pe kitna noise daalna hai, 0.0001 se 0.02 tak
        self.betas = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alphas = 1.0 - self.betas
        # alpha_bar[t] = alpha[0] * alpha[1] * ... * alpha[t]
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def add_noise(self, x0, t, noise=None):
        """x0 pe t step tak ka noise daal do. PDF wala formula:
        xt = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # t har image ke liye alag ho sakta hai
        sqrt_ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1m_ab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_1m_ab * noise
        return xt, noise

    def sample_step(self, model, xt, t):
        """Ek reverse step - thoda noise hatao."""
        with torch.no_grad():
            # Model batayega kitna noise hai
            pred_noise = model(xt, t)
            beta = self.betas[t].view(-1, 1, 1, 1)
            alpha = self.alphas[t].view(-1, 1, 1, 1)
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
            # Reverse formula - noise hatao
            mean = (1 / alpha.sqrt()) * (xt - (beta / (1 - alpha_bar).sqrt()) * pred_noise)
            if t[0].item() == 0:
                return mean
            noise = torch.randn_like(xt)
            return mean + beta.sqrt() * noise