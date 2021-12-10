import torch
import torch.nn as nn
from stg import StochasticGates
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from SDCCA import SparseDeepCCA

torch.manual_seed(555)


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
        try:
            xy = np.random.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, n, check_valid='raise')
        except:
            print('Cov not valid')
            not_valid = True

    x = xy[:, :p]
    y = xy[:, p:]
    return x, y, u, v, Sigma_xy


def plot_gates(net, name, u, v):
    g_x, g_y = net.get_gates()
    plt.plot(range(800), g_x.cpu().detach().numpy())
    plt.plot(range(800), u)
    plt.title(f'x gates,  {name}')
    plt.savefig(f'x_gates/x_gates_{name}.png')
    plt.close()
    plt.plot(range(800), g_y.cpu().detach().numpy())
    plt.plot(range(800), v)
    plt.title(f'y gates,  {name}')
    plt.savefig(f'y_gates/y_gates_{name}.png')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y, u, v, sigma_xy = gen_data(400, 800, 800)
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    lamx = 0.3
    lamy = 0.4
    net = SparseDeepCCA(x.shape[1], y.shape[1], [2048, 1024, 512, 4], [2048, 1024, 512, 4], lamx, lamy, x, y).to(device)
    net.train()
    optimizer = optim.Adam(net.parameters())
    plot_gates(net, f'{lamx}_{lamy}_0_(500,600,600,Topelitz)', u, v)
    for epoch in range(10000):
        optimizer.zero_grad()
        cor = net(x, y)
        cor.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}    loss: {cor}    lam: {lamx},{lamy}')
        if (epoch + 1) % 1000 == 0:
            plot_gates(net, f'{lamx}_{lamy}_{epoch+1}_(400,800,800,Linear)', u, v)


if __name__ == '__main__':
    main()

