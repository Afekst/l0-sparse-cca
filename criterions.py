import torch
import torch.nn as nn


def cov(x, y):
    x_bar = x - x.mean(axis=0)
    y_bar = y - y.mean(axis=0)
    N = x_bar.shape[0]
    return (y_bar.T @ x_bar).T / (N - 1)


class NegCorr(nn.Module):
    def __init__(self, device, eps=1e-5):
        super(NegCorr, self).__init__()
        self.device = device
        self.eps = eps

    def forward(self, x, y):
        C_yy = cov(y, y)
        C_yx = cov(y, x)
        C_xx = cov(x, x)

        C_yy = C_yy + torch.eye(C_yy.shape[0], device=self.device) * self.eps
        C_xx = C_xx + torch.eye(C_xx.shape[0], device=self.device) * self.eps

        M = torch.linalg.multi_dot([torch.inverse(C_yy),
                                    C_yx,
                                    torch.inverse(C_xx),
                                    C_yx.T])
        return -torch.trace(M)
