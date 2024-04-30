import torch
import torch.nn as nn
from PaRet import ParallelRetention
from MGD import MultiReDiffusion


class MGDPR(nn.Module):
    def __init__(self, diffusion, retention, ret_linear_1, ret_linear_2, post_pro, layers, num_nodes, time_dim, num_relation, gamma,
                 expansion_steps):

        super(MGDPR, self).__init__()
        self.layers = layers

        # Initialize transition matrices, weight coefficients, causal masking and exponential decay matrix for all layers
        self.T = nn.Parameter(torch.randn(layers, num_relation, expansion_steps, num_nodes, num_nodes))
        self.theta = nn.Parameter(torch.randn(layers, num_relation, expansion_steps))
        lower_tri = torch.tril(torch.ones(time_dim, time_dim), diagonal=-1)
        self.D_gamma = torch.where(lower_tri == 0, torch.tensor(0.0), gamma ** -lower_tri)

        # Initialize different module layers
        self.diffusion_layers = nn.ModuleList(
            [MultiReDiffusion(diffusion[i], diffusion[i + 1], num_relation) for i in range(len(diffusion) - 1)])

        self.retention_layers = nn.ModuleList(
            [ParallelRetention(time_dim, retention[3 * i], retention[3 * i + 1], retention[3 * i + 2]) for i in
             range(len(retention) // 3)]
        )
        self.ret_linear_1 = nn.ModuleList(
            [nn.Linear(ret_linear_1[2 * i], ret_linear_1[2 * i + 1]) for i in range(len(ret_linear_1) // 2)])

        self.ret_linear_2 = nn.ModuleList(
            [nn.Linear(ret_linear_2[2 * i], ret_linear_2[2 * i + 1]) for i in range(len(ret_linear_2) // 2)])

        self.mlp = nn.ModuleList([nn.Linear(post_pro[i], post_pro[i + 1]) for i in range(len(post_pro) - 1)])


    def forward(self, x, a):
        # Initialize h with x
        h = x

        # Information diffusion and graph representation learning
        for l in range(self.layers):

            # Multi-relational Graph Diffusion
            h, u = self.diffusion_layers[l](self.theta[l], self.T[l], a, h)

            # Parallel Retention
            eta = self.retention_layers[l](u.to('cuda'), self.D_gamma)

            # Decoupled representation transform
            if l == 0:
                h_prime = torch.concat((eta, self.ret_linear_1[l](x.view(x.shape[1], -1))), dim=1)
                h_prime = self.ret_linear_2[l](h_prime)
            else:
                h_prime = torch.concat((eta, self.ret_linear_1[l](h_prime)), dim=1)
                h_prime = self.ret_linear_2[l](h_prime)

        # Post-processing to generate final graph representation
        for mlp in self.mlp:
            h_prime = mlp(h_prime)

        return h_prime

    def reset_parameters(self):
        """
        Reset model parameters with appropriate initialization methods.
        """
        nn.init.normal_(self.T)
        nn.init.normal_(self.D_gamma)
        nn.init.normal_(self.theta)
