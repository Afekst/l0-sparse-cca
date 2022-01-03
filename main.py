import torch
import torch.nn as nn
import torch.nn.functional as F
from stg import StochasticGates
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from SDCCA import SparseDeepCCA
import faulthandler; faulthandler.enable()



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


def plot_gates(net, name, u=None, v=None):
    g_x, g_y = net.get_gates()
    g_x = g_x.cpu().detach().numpy()
    np.save('gates.npy', g_x)
    plt.imshow(np.reshape(g_x, (110, 136)))
    plt.colorbar()
    if u is not None:
        plt.plot(range(800), u)
    plt.title(f'x gates,  {name}')
    plt.savefig(f'x_gates/x_gates_{name}.png')
    plt.close()
    plt.plot(g_y.cpu().detach().numpy())
    if v is not None:
        plt.plot(range(800), v)
    plt.title(f'y gates,  {name}')
    plt.savefig(f'y_gates/y_gates_{name}.png')
    plt.close()
    
def train(net, optimizer, x, y):
    optimizer.zero_grad()
    loss = net(x, y)
    loss.backward()
    optimizer.step()
    return loss
    
    
    


def main(args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    x, y, u, v, sigma_xy = gen_data(400, 800, 800, flag=2)
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    #x = np.load('bennet_flow.npy')
    #y = np.load('bennet_audio.npy').T
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    lamx = 0.05
    lamy = 0.05
    net = SparseDeepCCA(x.shape[1], y.shape[1], [4096, 2048, 1024, 1024, 1024, 64], [4096, 2048, 1024, 64], lamx, lamy, device)
    net = net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters())
    #plot_gates(net, f'{lamx}_{lamy}_0_bennet')
    loss=[]
    for epoch in range(50000):
        loss.append(train(net, optimizer, x, y).item())
        if (epoch + 1) % 100 == 0:
            print(f'epoch: {epoch + 1}    loss: {loss[-1]}    lam: {lamx},{lamy}')
        if (epoch + 1) % 1000 == 0:
            plot_gates(net, f'{lamx}_{lamy}_{epoch+1}_bennet')
    plt.plot(loss)
    plt.savefig('loss.png')
    plt.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cuda',
                        help='gpu index',
                        type=str,
                        default="0")
    args = parser.parse_args()
    main(args)

