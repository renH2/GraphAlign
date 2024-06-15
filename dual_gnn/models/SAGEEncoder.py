import torch.nn as nn
import torch
import scipy.stats as stats
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels),
            SAGEConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels),
            nn.PReLU(hidden_channels)
        ])

    def forward(self, x, adjs):
        print(x)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


def weighted_mean(x, w):
    return torch.sum(w * x) / torch.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters,
                 alphas_init,
                 betas_init,
                 weights_init):
        self.alphas = alphas_init
        self.betas = betas_init
        self.weight = weights_init
        self.max_iters = max_iters
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        x_cpu = x.cpu().detach().numpy()
        alpha_cpu = self.alphas.cpu().detach().numpy()
        beta_cpu = self.betas.cpu().detach().numpy()
        return torch.from_numpy(stats.beta.pdf(x_cpu, alpha_cpu[y], beta_cpu[y])).to(x.device)

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return self.weighted_likelihood(x, 0) + self.weighted_likelihood(x, 1)

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = torch.cat((self.weighted_likelihood(x, 0).view(1, -1), self.weighted_likelihood(x, 1).view(1, -1)), 0)
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(0)
        return r

    def fit(self, x):
        eps = 1e-12
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            if self.betas[1] < 1:
                self.betas[1] = 1.01
            self.weight = r.sum(1)
            self.weight /= self.weight.sum()
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
