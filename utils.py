import torch.nn as nn
from prettytable import PrettyTable


class XNet(nn.Module):
    def __init__(self):
        super(XNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(6, 8, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*25*20, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class YNet(nn.Module):
    def __init__(self, in_dim):
        super(YNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def print_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
