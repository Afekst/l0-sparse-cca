import torch
import torch.nn as nn
import math


class StochasticGates(nn.Module):
    def __init__(self, size, sigma, lam, gate_init=None):
        super().__init__()
        self.size = size
        if gate_init is None:
            mus = 1.0 * torch.ones(size)
        else:
            mus = torch.from_numpy(gate_init)
        self.mus = nn.Parameter(mus, requires_grad=True)
        self.sigma = sigma
        self.lam = lam

    def forward(self, x):
        gaussian = self.sigma * torch.randn(self.mus.size()) * self.training
        shifted_gaussian = self.mus + gaussian.to(x.device)
        z = torch.clamp(shifted_gaussian, 0.0, 1.0)
        new_x = x * z
        return new_x

    def get_reg(self):
        return self.lam * torch.sum((1 + torch.erf((self.mus / self.sigma) / math.sqrt(2))) / 2)

    def get_gates(self):
        return torch.clamp(self.mus, 0.0, 1.0)
