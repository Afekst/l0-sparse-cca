import torch
import torch.nn as nn
import numpy as np
from stg import StochasticGates

class SparseDeepCCA(nn.Module):
    def __init__(self, x_features, y_features, x_architecture, y_architecture, lamx, lamy, device, X=None, Y=None):
        """
        c'tor to l0-DCCA class
        :param x_features: Dx
        :param y_features: Dy
        :param x_architecture: python list of neurons in each fully-connected layer for X modality
        :param y_architecture: python list of neurons in each fully-connected layer for Y modality
        :param lamx: regularizer for X gates
        :param lamy: regularizer for Y gates
        :param X: samples used for gates initialization
        :param Y: samples used for gates initialization
        """
        super().__init__()
        if x_architecture[-1] != y_architecture[-1]:
            raise ValueError('final layer in each network must be the same size')
        if bool(X is None) != bool(Y is None):
            raise ValueError('for gate initialization, you must provide both X and Y.')
        self.device = device
        x_gates, y_gates = self.gates_init(X, Y, 100)
        self.f = self._create_network(x_features, x_architecture, lamx, x_gates)
        self.g = self._create_network(y_features, y_architecture, lamy, y_gates)
        

    def forward(self, X, Y):
        """
        forward pass in l0-DCCA
        :param X: 1st modality N (samples) x D_x (features)
        :param Y: 2nd modality N (samples) x D_y (features)
        :return: if eval then the output of each non-linear function, if train then return the loss function
        """
        X_hat, Y_hat = self.f(X), self.g(Y)
        if not self.training:
            return self.f[-1].weight, self.g[-1].weight
        else:
            return -self._get_corr(X_hat, Y_hat) + self.f[1].get_reg() + self.g[1].get_reg()

    @staticmethod
    def gates_init(X, Y, k):
        """
        calculate initial guess for gates value based on CCA
        :param X: samples used for gates initialization
        :param Y: samples used for gates initialization
        :param k: number of active gates
        :return: initial guess for the gates
        """
        if X is None or Y is None:
            return None, None
        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        N = X.shape[0]
        C_xy = (X.T @ Y)/(N - 1)
        thresh = np.percentile(np.abs(C_xy.reshape(-1)), 100 * (1 - k / X.shape[1]))
        C_xy[np.abs(C_xy) < thresh] = 0
        [U0, s, V0] = np.linalg.svd(C_xy)
        uu = np.abs(U0[:, 0])
        uu[uu < 1 * thresh] -= 1.5
        vv = np.abs(V0[0, :])
        vv[vv < 1 * thresh] -= 1.5

        return uu, vv


    def get_gates(self):
        return self.f[1].get_gate(), self.g[1].get_gate()

    def _create_network(self, in_features, architecture, lam, gates=None):
        """
        create a sequential network with STG
        :param in_features: number of features in th input to the network
        :param architecture: python list where each element is thh number on neurons in the layer
        :param lam: regularizer for the STG
        :return: sequential network with STG, and len(architecture) layers
        """
        layers = nn.ModuleList([])
        layers.append(nn.BatchNorm1d(in_features))
        layers.append(StochasticGates(in_features, 1, lam, self.device, gates))
        for i in range(len(architecture)):
            if i == 0:
                layers.append(nn.Linear(in_features, architecture[i]))
                layers.append(nn.Tanh())
            elif i == len(architecture)-1:
                layers.append(nn.Linear(architecture[i-1], architecture[i]))
            else:
                layers.append(nn.Linear(architecture[i-1], architecture[i]))
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

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

        C_yy_inv_root = self._mat_to_the_power(C_yy+torch.eye(C_xx.shape[0], device=self.device)*1e-3, -0.5)  # DATA MUST BE SPARSE!
        C_xx_inv = torch.inverse(C_xx+torch.eye(C_xx.shape[0], device=self.device)*1e-3)
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
