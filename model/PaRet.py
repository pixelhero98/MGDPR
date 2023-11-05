import torch
import torch.nn as nn  # Added import for nn

class ParallelRetention(torch.nn.Module):

    def __init__(self, time_dim, in_dim, inter_dim, out_dim):
        super(ParallelRetention, self).__init__()
        self.time_dim = time_dim
        self.in_dim = in_dim
        self.inter_dim = inter_dim
        self.out_dim = out_dim

        self.activation = torch.nn.LeakyReLU()

        self.Q_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.K_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.V_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.ret_feature = torch.nn.Linear(self.inter_dim, self.out_dim)


    def forward(self, x, d_gamma):

        num_node = x.shape[1]
        x = x.permute(2, 1, 0).contiguous().view(self.time_dim, -1)
        x0 = self.Q_layers(x) @ self.K_layers(x).transpose(0, 1)
        x = (d_gamma * x0) @ self.V_layers(x)
        x = self.activation(self.ret_feature(x))

        return x.view(num_node, -1)
