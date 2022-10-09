import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def gen_data(n=400, p=800, q=800, rho=0.9, flag=0, fac=1):
    Sigma_x = 1 * np.eye(p)
    Sigma_y = 1 * np.eye(q)
    Sigma_0 = 1 * np.eye(p)
    if flag == 1:
        for i in range(p):
            for j in range(p):
                Sigma_x[i, j] = 0.9 ** (abs(i - j))
    elif flag == 2:
        for i in range(p):
            for j in range(p):
                if np.abs(i - j) == 1:
                    Sigma_0[i, j] = 0.5
                elif np.abs(i - j) == 2:
                    Sigma_0[i, j] = 0.4
        Sigma_1 = np.copy(np.linalg.inv(Sigma_0))
        Sigma_x = np.copy(Sigma_1)
        for i in range(p):
            for j in range(p):
                Sigma_x[i, j] = Sigma_1[i, j] / (np.sqrt(Sigma_1[i, i] * Sigma_1[j, j]))

    Sigma_y = Sigma_x

    k = 5
    not_valid = True
    while not_valid:
        u = np.zeros((p, 1))
        v = np.zeros((q, 1))
        ind1 = np.random.choice(np.arange(p), k)
        ind2 = np.random.choice(np.arange(q), k)
        u[ind1] = fac / np.sqrt(k)
        v[ind2] = fac / np.sqrt(k)
        Sigma_xy = rho * Sigma_x @ u @ v.T @ Sigma_y
        Sigma1 = np.hstack((Sigma_x, Sigma_xy))
        Sigma2 = np.hstack((Sigma_xy.T, Sigma_y))
        Sigma = np.vstack((Sigma1, Sigma2))
        #    pos_def=is_pos_def(Sigma)
        #   if pos_def:
        not_valid = False
        if sum(u != 0) != k or sum(v != 0) != k:
            print('problems with u or v')
            not_valid = True
        try:
            xy = np.random.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, n, check_valid='raise')
        except:
            print('Cov not valid')
            not_valid = True

    x = xy[:, :p]
    y = xy[:, p:]
    return x, y, u, v, Sigma_xy


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        return nn.functional.normalize(x, dim=-1)


def plot_gates(net, name, u, v):
    g_x, g_y = net.get_gates()
    g_x = g_x.cpu().detach().numpy().T
    g_y = g_y.cpu().detach().numpy().T

    plt.subplot(2, 1, 1)
    plt.stem(u, basefmt=" ", markerfmt='D')
    plt.stem(g_x, basefmt=" ", linefmt='r')
    plt.title(f'x gates')

    plt.subplot(2, 1, 2)
    plt.stem(v, basefmt=" ", markerfmt='D')
    plt.stem(g_y, basefmt=" ", linefmt='r')
    plt.title(f'y gates')

    plt.suptitle(f'{name}')
    plt.show()
    plt.close()





