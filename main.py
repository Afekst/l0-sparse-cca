import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from stg import StochasticGates
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from SDCCA import SparseDeepCCA
import faulthandler; faulthandler.enable()  # for segmentation fault debugging
from timeit import default_timer as timer
import utils
from utils import XNet, YNet
from scipy.io import loadmat

testing_on_cpu = False


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


def train_step(net, funcs_opt, gates_opt, x, y):
    funcs_opt.zero_grad()
    gates_opt.zero_grad()
    loss = net(x, y).mean()
    loss.backward()
    funcs_opt.step()
    gates_opt.step()
    return loss


def plot_gates(net, name):
    if testing_on_cpu:
        return
    g_x, g_y = net.module.get_gates()
    g_x = g_x.cpu().detach().numpy().T
    np.save('gates.npy', g_x)
    #plt.imshow(np.reshape(g_x, (340, 270)))
    plt.imshow(g_x)
    plt.colorbar()
    plt.title(f'x gates,  {name}')
    plt.savefig(f'x_gates/x_gates_{name}.png')
    plt.close()
    plt.plot(g_y.cpu().detach().numpy())
    plt.title(f'y gates,  {name}')
    plt.savefig(f'y_gates/y_gates_{name}.png')
    plt.close()




def main(args):
    device = torch.device(f'cuda:{args.cuda[0]}' if torch.cuda.is_available() else 'cpu')
    if testing_on_cpu:
        x, y, u, v, sigma_xy = gen_data(400, 800, 800, flag=2)
    else:
        x = loadmat('matlab_try/flow_2.mat')['out']
        x = np.expand_dims(x, axis=2)
        x = x.T
        print(x.shape)
        y = np.load('vad_audio.npy').T
        print(y.shape)
    lamx = args.lamx
    lamy = args.lamy
    x_net = XNet()
    y_net = YNet(y.shape[1])
    net = SparseDeepCCA([x.shape[2], x.shape[3]], y.shape[1], x_net, y_net, lamx, lamy, 1, 1)
    utils.print_parameters(net)
    if torch.cuda.is_available():
        net = nn.DataParallel(net, device_ids=args.cuda)
    else:
        net = nn.DataParallel(net)  # because nn.DataParallel wraps nn.Module
    net = net.to(device)
    net.train()
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    funcs_params = net.module.get_function_parameters()
    gates_params = net.module.get_gates_parameters()
    funcs_opt = optim.Adam(funcs_params, lr=1e-4)
    gates_opt = optim.Adam(gates_params, lr=1e-3)
    plot_gates(net, f'{lamx}_{lamy}_0_vad_2')
    loss = []
    start = timer()
    for epoch in range(100000):
        loss.append(train_step(net, funcs_opt, gates_opt, x, y).item())
        if (epoch + 1) % 100 == 0:
            end = timer()
            print(f'epoch: {epoch + 1}    loss: {loss[-1]:.4f}    lam: {lamx},{lamy}    time: {end-start:.2f}')
            start = end
        if (epoch + 1) % 1000 == 0:
            plot_gates(net, f'{lamx}_{lamy}_{epoch+1}_vad_2')
    plt.plot(loss)
    plt.savefig('loss_2.png')
    plt.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cuda',
                        help='gpu indexes, in [1,2,3] format',
                        type=str,
                        default="[1,2,3]")
    parser.add_argument('--lamx',
                        help='x gates reg',
                        type=float,
                        default=1.0)
    parser.add_argument('--lamy',
                        help='y gates reg',
                        type=float,
                        default=1.0)
    args = parser.parse_args()

    args.cuda = args.cuda.strip('][').split(',')
    args.cuda = [int(e) for e in args.cuda]
    main(args)

