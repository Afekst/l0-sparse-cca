import torch.nn as nn
from STG import StochasticGates


class SDCCA(nn.Module):
    def __init__(self, x_size, f, lamx, sigmax,
                       y_size, g, lamy, sigmay):
        super().__init__()
        self.gated_f = nn.Sequential(
            StochasticGates(x_size, sigmax, lamx),
            f)
        self.gated_g = nn.Sequential(
            StochasticGates(y_size, sigmay, lamy),
            g)

    def forward(self, X, Y):
        return self.gated_f(X), self.gated_g(Y)

    def get_reg(self):
        return self.gated_f[0].get_reg().mean() + self.gated_g[0].get_reg().mean()

    def get_gates(self):
        """
        use this function to retrieve the gates values for each modality
        :return: gates values
        """
        return self.gated_f[0].get_gates(), self.gated_g[0].get_gates()

    def get_function_parameters(self):
        """
        use this function if you wish to use a different optimizer for functions and gates
        :return: learnable parameters of f and g
        """
        params = list()
        for function in [self.gated_f, self.gated_g]:
            params += list(function[1].parameters())
        return params

    def get_gates_parameters(self):
        """
        use this function if you wish to use a different optimizer for functions and gates
        :return: learnable parameters of the gates
        """
        params = list()
        for function in [self.gated_f, self.gated_g]:
            params += list(function[0].parameters())
        return params
