import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from SDCCA import SparseDeepCCA
from timeit import default_timer as timer
import utils
from utils import XNet, YNet
from scipy.io import loadmat


def train_step(net, funcs_opt, gates_opt, x, y):
    funcs_opt.zero_grad()
    gates_opt.zero_grad()
    loss = net(x, y).mean()
    loss.backward()
    funcs_opt.step()
    gates_opt.step()
    return loss


def plot_gates(net, name):
    g_x, g_y = net.module.get_gates()
    g_x = g_x.cpu().detach().numpy().T
    np.save('gates.npy', g_x)
    plt.imshow(g_x)
    plt.colorbar()
    plt.title(f'x gates,  {name}')
    plt.savefig(f'x_gates/x_gates_{name}.png')
    plt.close()
    plt.plot(g_y.cpu().detach().numpy())
    plt.title(f'y gates,  {name}')
    plt.savefig(f'y_gates/y_gates_{name}.png')
    plt.close()


def load_data():
    x = loadmat('matlab_try/flow_2.mat')['out']
    x = np.expand_dims(x, axis=2)
    x = x.T
    y = np.load('vad_audio.npy').T
    return x, y


def main(args):
    device = torch.device(f'cuda:{args.cuda[0]}' if torch.cuda.is_available() else 'cpu')
    x, y = load_data()
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)

    x_net = XNet()
    y_net = YNet(y.shape[1])
    net = SparseDeepCCA([x.shape[2], x.shape[3]], y.shape[1], x_net, y_net, args.lamx, args.lamy)
    utils.print_parameters(net)

    if torch.cuda.is_available():
        net = nn.DataParallel(net, device_ids=args.cuda)
    else:
        net = nn.DataParallel(net)  # because nn.DataParallel wraps nn.Module

    net = net.to(device)
    net.train()

    funcs_params = net.module.get_function_parameters()
    gates_params = net.module.get_gates_parameters()
    funcs_opt = optim.Adam(funcs_params, lr=1e-4)
    gates_opt = optim.Adam(gates_params, lr=1e-3)

    plot_gates(net, f'{args.lamx}_{args.lamy}_0_vad')
    loss = []
    start = timer()
    for epoch in range(100000):
        loss.append(train_step(net, funcs_opt, gates_opt, x, y).item())
        if (epoch + 1) % 100 == 0:
            end = timer()
            print(f'epoch: {epoch + 1}    '
                  f'loss: {loss[-1]:.4f}    '
                  f'lam: {args.lamx}, {args.lamy}    '
                  f'time: {end-start:.2f}')
            start = end
        if (epoch + 1) % 1000 == 0:
            plot_gates(net, f'{args.lamx}_{args.lamy}_{epoch+1}_vad')
    plt.plot(loss)
    plt.savefig('loss.png')
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

