from SDCCA import SDCCA
from utils import plot_gates, gen_data, L2Norm
from criterions import NegCorr
import torch.nn as nn
import torch.optim as optim
import torch


def create_networks(lamx, sigmax, lamy, sigmay, size):
    f = nn.Sequential(nn.Linear(size, 100, bias=False),
                      nn.ReLU(),
                      nn.Linear(100, 100, bias=False),
                      nn.ReLU(),
                      nn.Linear(100, 10, bias=False),
                      L2Norm())

    g = nn.Sequential(nn.Linear(size, 100, bias=False),
                      nn.ReLU(),
                      nn.Linear(100, 100, bias=False),
                      nn.ReLU(),
                      nn.Linear(100, 10, bias=False),
                      L2Norm())

    net = SDCCA(size, f, lamx, sigmax,
                size, g, lamy, sigmay)

    return net


def train(net, criterion, x, y, gates_optim, func_optim, k):
    net.train()
    if k == 0:
        gates_optim.zero_grad()
    func_optim.zero_grad()

    x_emb, y_emb = net(x, y)
    neg_corr = criterion(x_emb, y_emb)
    reg = net.get_reg()
    loss = neg_corr + reg
    loss.backward()

    func_optim.step()
    if k == 0:
        gates_optim.step()

    return neg_corr.item(), reg.item()


def simple_train(net, criterion, x, y, optimizer):
    net.train()
    optimizer.zero_grad()

    x_emb, y_emb = net(x, y)
    neg_corr = criterion(x_emb, y_emb)
    reg = net.get_reg()
    loss = neg_corr + reg
    loss.backward()

    optimizer.step()

    return neg_corr.item(), reg.item()


def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    lamx = 5e-2
    lamy = 1e-5
    sigmax = 1.0
    sigmay = 1.0
    lamy = lamx  # TODO: remove after tests

    samples = 1000
    size = 20

    epochs = 50000
    K = 10

    x, y, u, v, _ = gen_data(samples, size, size)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    y = torch.tensor(y, device=device, dtype=torch.float32)

    net = create_networks(lamx, sigmax, lamy, sigmay, size).to(device)

    criterion = NegCorr(device)
    gates_optimizer = optim.Adam(net.get_gates_parameters(), lr=1e-1)
    funcs_optimizer = optim.Adam(net.get_function_parameters(), lr=1e-4)
    for epoch in range(epochs):
        neg_corr, reg = train(net, criterion, x, y, gates_optimizer, funcs_optimizer, (epoch+1)%K)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch[{epoch+1}/{epochs}]: corr: {-neg_corr}, reg: {reg}')
        if (epoch + 1) % 1000 == 0:
            plot_gates(net, epoch+1, u, v)


if __name__ == '__main__':
    main()

