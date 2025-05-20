import torch
import torch.nn as nn

class ParallelRetention(torch.nn.Module):

    def __init__(self, time_dim, in_dim, inter_dim, out_dim):
        super(ParallelRetention, self).__init__()
        self.time_dim = time_dim
        self.in_dim = in_dim
        self.inter_dim = inter_dim
        self.out_dim = out_dim
        self.activation = torch.nn.PReLU()
        self.Q_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.K_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.V_layers = nn.Linear(self.in_dim, self.inter_dim)
        self.ret_feat = torch.nn.Linear(self.inter_dim, self.out_dim)
        # Group normalization as described in the paper
        self.group_norm = nn.GroupNorm(1, self.inter_dim)

    def forward(self, x, d_gamma):
        num_node = x.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d_gamma = d_gamma.to(device)
        x = x.view(self.time_dim, -1)

        inter_feat = self.Q_layers(x) @ self.K_layers(x).transpose(0, 1)
        x = (d_gamma * inter_feat) @ self.V_layers(x)
        # Apply group normalization before non-linearity
        x = x.transpose(0, 1).unsqueeze(0)
        x = self.group_norm(x)
        x = x.squeeze(0).transpose(0, 1)
        x = self.activation(self.ret_feat(x))

        return x.view(num_node, -1)
