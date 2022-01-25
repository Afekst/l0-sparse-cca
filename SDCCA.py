import torch
import torch.nn as nn
import numpy as np
from stg import StochasticGates

class SparseDeepCCA(nn.Module):
    def __init__(self, x_dim, y_dim, x_net, y_net, lamx, lamy):
        """
        c'tor to l0-DCCA class
        :param x_features: Dx
        :param y_features: Dy
        :param x_architecture: python list of neurons in each fully-connected layer for X modality
        :param y_architecture: python list of neurons in each fully-connected layer for Y modality
        :param lamx: regularizer for X gates
        :param lamy: regularizer for Y gates
        """
        super().__init__()
        self.f = self._create_network(x_dim, x_net, lamx)
        self.g = self._create_network(y_dim, y_net, lamy)
        

    def forward(self, X, Y):
        """
        forward pass in l0-DCCA
        :param X: 1st modality N (samples) x D_x (features)
        :param Y: 2nd modality N (samples) x D_y (features)
        :return: if eval then the output of each non-linear function, if train then return the loss function
        """
        X_hat, Y_hat = self.f(X), self.g(Y)
        return -self._get_corr(X_hat, Y_hat) + self.f[0].get_reg() + self.g[0].get_reg()

    def get_gates(self):
        return self.f[0].get_gates(), self.g[0].get_gates()

    def get_function_parameters(self):
        params = list()
        for net in [self.f, self.g]:
            params += list(net[1].parameters())
        return params

    def get_gates_parameters(self):
        params = list()
        for net in [self.f, self.g]:
            params += list(net[0].parameters())
        return params


    @staticmethod
    def _create_network(in_features, net, lam):
        return nn.Sequential(StochasticGates(in_features, 1, lam),
                             net)

    def _get_corr(self, X, Y):
        """
        computes the correlation between X,Y
        :param X: 1st variable, N (samples) x d (features)
        :param Y: 2nd variable, N (samples) x d (features)
        :return: rho(X,Y)
        """
        psi_x = X - X.mean(axis=0)
        psi_y = Y - Y.mean(axis=0)

        C_yy = self._cov(psi_y, psi_y)
        C_yx = self._cov(psi_y, psi_x)
        C_xy = self._cov(psi_x, psi_y)
        C_xx = self._cov(psi_x, psi_x)

        C_yy_inv_root = self._mat_to_the_power(C_yy+torch.eye(C_yy.shape[0], device=Y.device)*1e-3, -0.5)
        C_xx_inv = torch.inverse(C_xx+torch.eye(C_xx.shape[0], device=X.device)*1e-3)
        M = torch.linalg.multi_dot([C_yy_inv_root, C_yx, C_xx_inv, C_xy, C_yy_inv_root])
        return torch.trace(M)/M.shape[0]

    @staticmethod
    def _cov(psi_x, psi_y):
        """
        estimates the covariance matrix between two centered views
        :param psi_x: 1st centered view, N (samples) x d (features)
        :param psi_y: 2nd centered view, N (samples) x d (features)
        :return: covariance matrix
        """
        N = psi_x.shape[0]
        return (psi_y.T @ psi_x).T / (N - 1)

    @staticmethod
    def _mat_to_the_power(A, arg):
        """
        raises matrix to the arg-th power using diagonalization, where arg is signed float.
        if arg is integer, it's better to use 'torch.linalg.matrix_power()'
        :param A: symmetric matrix (must be PSD if taking even roots)
        :param arg: the power
        :return: A^(arg)
        """
        eig_values, eig_vectors = torch.linalg.eig(A)
        eig_values = torch.real(eig_values)
        eig_vectors = torch.real(eig_vectors)
        return torch.linalg.multi_dot([eig_vectors, torch.diag((eig_values+1e-3) ** arg), torch.inverse(eig_vectors)])
